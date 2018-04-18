using UnityEngine;

public static class GlobalInfo
{
    public static bool IsFirst { get; set; }
    public static string HostIP { get; set; }
    public static int HostPort { get; set; }
    public static Object ReadLock = new Object();
}