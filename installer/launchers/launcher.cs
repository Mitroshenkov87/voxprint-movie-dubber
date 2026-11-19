using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

internal static class Program
{
    private static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
#if CONSOLE
        string pyName = "python.exe";
#else
        string pyName = "pythonw.exe";
#endif
        string py = Path.Combine(dir, "runtime", "Scripts", pyName);
        string main = Path.Combine(dir, "main.py");
        if (!File.Exists(py) || !File.Exists(main))
        {
            Fail("VOXPrint AI Movie Dubber could not find the shared Python runtime.\r\n" + py);
            return 1;
        }
        var start = new ProcessStartInfo();
        start.FileName = py;
        start.WorkingDirectory = dir;
        start.UseShellExecute = false;
        start.Arguments = Quote(main);
        for (int i = 0; i < args.Length; i++)
        {
            start.Arguments += " " + Quote(args[i]);
        }
        try
        {
            Process process = Process.Start(start);
            if (process == null)
            {
                Fail("VOXPrint AI Movie Dubber could not start.");
                return 1;
            }
            process.WaitForExit();
            return process.ExitCode;
        }
        catch (Exception ex)
        {
            Fail(ex.Message);
            return 1;
        }
    }

    private static void Fail(string message)
    {
        try
        {
            string log = Path.Combine(Path.GetTempPath(), "voxprint-movie-dubber-launch.log");
            File.AppendAllText(log, DateTime.Now.ToString("s") + " " + message + "\r\n");
        }
        catch (Exception)
        {
        }
#if CONSOLE
        Console.Error.WriteLine(message);
#else
        MessageBox(IntPtr.Zero, message, "VOXPrint AI Movie Dubber", 0x00000010);
#endif
    }

    private static string Quote(string value)
    {
        if (value.IndexOfAny(new char[] { ' ', '\t', '\n', '\v', '"' }) < 0)
        {
            return value;
        }
        var builder = new StringBuilder();
        builder.Append('"');
        int slashes = 0;
        for (int i = 0; i < value.Length; i++)
        {
            char ch = value[i];
            if (ch == '\\')
            {
                slashes++;
                continue;
            }
            if (ch == '"')
            {
                builder.Append('\\', slashes * 2 + 1);
                builder.Append(ch);
                slashes = 0;
                continue;
            }
            if (slashes > 0)
            {
                builder.Append('\\', slashes);
                slashes = 0;
            }
            builder.Append(ch);
        }
        if (slashes > 0)
        {
            builder.Append('\\', slashes * 2);
        }
        builder.Append('"');
        return builder.ToString();
    }

#if !CONSOLE
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr owner, string text, string caption, uint type);
#endif
}
