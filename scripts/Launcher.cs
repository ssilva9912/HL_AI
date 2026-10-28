using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;

// Starts Docker Desktop, Ollama and the Homelab AI stack, then opens the app in the browser.
// Expects to live in <project>\dist so the project root (and compose.yaml) is one level up.
internal static class Launcher
{
    private const string AppUrl = "http://127.0.0.1:8501";
    private const string OllamaUrl = "http://127.0.0.1:11434";

    private static readonly string DockerDesktop =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Docker\Docker\Docker Desktop.exe");
    private static readonly string DockerCli =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Docker\Docker\resources\bin\docker.exe");
    private static readonly string OllamaApp =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Ollama\ollama app.exe");

    private static readonly string VsCode =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Microsoft VS Code\Code.exe");

    private static Label statusLabel;

    [STAThread]
    private static void Main()
    {
        string projectRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".."));
        if (!File.Exists(Path.Combine(projectRoot, "compose.yaml")))
        {
            MessageBox.Show(
                "Could not find compose.yaml in:\n" + projectRoot + "\n\nKeep HomelabAI.exe inside the project's dist folder, or rebuild it with scripts\\build-launcher.ps1.",
                "Homelab AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        Application.EnableVisualStyles();
        Form form = new Form();
        form.Text = "Homelab AI";
        form.ClientSize = new Size(380, 80);
        form.StartPosition = FormStartPosition.CenterScreen;
        form.FormBorderStyle = FormBorderStyle.FixedDialog;
        form.ControlBox = false;
        form.TopMost = true;
        statusLabel = new Label();
        statusLabel.Dock = DockStyle.Fill;
        statusLabel.TextAlign = ContentAlignment.MiddleCenter;
        statusLabel.Font = new Font("Segoe UI", 10);
        form.Controls.Add(statusLabel);

        form.Shown += delegate
        {
            Thread worker = new Thread(delegate()
            {
                string error = null;
                try { Launch(projectRoot); }
                catch (Exception ex) { error = ex.Message; }
                form.BeginInvoke((MethodInvoker)delegate
                {
                    form.Hide();
                    if (error != null)
                    {
                        MessageBox.Show(error, "Homelab AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    form.Close();
                });
            });
            worker.IsBackground = true;
            worker.Start();
        };
        Application.Run(form);
    }

    private static void Launch(string projectRoot)
    {
        OpenInVsCode(projectRoot);

        SetStatus("Checking Docker...");
        if (!DockerReady())
        {
            if (!File.Exists(DockerDesktop))
            {
                throw new Exception("Docker Desktop was not found at:\n" + DockerDesktop);
            }
            SetStatus("Starting Docker Desktop (this can take a minute)...");
            Process.Start(DockerDesktop);
            WaitUntil(DockerReady, 240, "Docker Desktop did not become ready within 4 minutes.");
        }

        if (!UrlResponds(OllamaUrl))
        {
            if (!File.Exists(OllamaApp))
            {
                throw new Exception("Ollama was not found at:\n" + OllamaApp);
            }
            SetStatus("Starting Ollama...");
            Process.Start(OllamaApp);
            WaitUntil(delegate { return UrlResponds(OllamaUrl); }, 60, "Ollama did not respond on " + OllamaUrl + ".");
        }

        SetStatus("Starting Homelab AI containers...");
        if (RunDocker("compose up -d", projectRoot) != 0)
        {
            throw new Exception("'docker compose up -d' failed. Run it in a terminal from the project folder to see the error.");
        }

        SetStatus("Waiting for the app to come up...");
        WaitUntil(delegate { return UrlResponds(AppUrl + "/_stcore/health"); }, 180, "The app did not respond on " + AppUrl + ".");

        Process.Start(new ProcessStartInfo(AppUrl) { UseShellExecute = true });
    }

    // Best effort: VS Code is a convenience, so a missing install never blocks the app.
    private static void OpenInVsCode(string projectRoot)
    {
        if (!File.Exists(VsCode))
        {
            return;
        }

        try
        {
            Process.Start(new ProcessStartInfo(VsCode, "\"" + projectRoot + "\"") { UseShellExecute = false });
        }
        catch (Exception)
        {
        }
    }

    private static void SetStatus(string text)
    {
        statusLabel.BeginInvoke((MethodInvoker)delegate { statusLabel.Text = text; });
    }

    private static void WaitUntil(Func<bool> condition, int timeoutSeconds, string failure)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
        while (DateTime.UtcNow < deadline)
        {
            if (condition()) { return; }
            Thread.Sleep(1000);
        }
        throw new Exception(failure);
    }

    private static bool DockerReady()
    {
        return RunDocker("info", null) == 0;
    }

    private static int RunDocker(string arguments, string workingDirectory)
    {
        ProcessStartInfo info = new ProcessStartInfo(File.Exists(DockerCli) ? DockerCli : "docker", arguments);
        info.UseShellExecute = false;
        info.CreateNoWindow = true;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        if (workingDirectory != null) { info.WorkingDirectory = workingDirectory; }
        try
        {
            using (Process process = Process.Start(info))
            {
                process.StandardOutput.ReadToEnd();
                process.StandardError.ReadToEnd();
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return -1;
        }
    }

    private static bool UrlResponds(string url)
    {
        try
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
            request.Timeout = 2000;
            using (request.GetResponse()) { return true; }
        }
        catch (Exception)
        {
            return false;
        }
    }
}
