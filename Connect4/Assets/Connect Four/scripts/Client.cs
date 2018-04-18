using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class Client
{
    private int myPort;
    private bool isRunning = true;
    private Thread clientThread;
    private UdpClient ReceiveUDPClient;
    public string HostIP { get; set; }
    public int HostPort { get; set; }
    public Action<String> MessageHandler { get; set; }

    public Client(string HostIP, int port, Action<string> method)
    {
        this.HostIP = HostIP;
        this.HostPort = port;
        this.MessageHandler = method;
        myPort = 11111;

        clientThread = new Thread(Listen);
        clientThread.Start();
    }

    public void Close()
    {
        isRunning = false;
        try
        {
            ReceiveUDPClient.Close();
        }
        catch (Exception ex)
        {
            Debug.Log(ex);
        }
    }


    public void Listen()
    {
        try
        {
            while (isRunning)
            {
                Debug.Log("listening," + isRunning.ToString());
                using (ReceiveUDPClient = new UdpClient(myPort+1))
                {
                    //IPEndPoint object will allow us to read datagrams sent from any source.
                    IPEndPoint RemoteIpEndPoint = new IPEndPoint(IPAddress.Parse(HostIP), HostPort);
                    // Blocks until a message returns on this socket from a remote host.
                    Debug.Log("test0");
                    Byte[] receiveBytes = ReceiveUDPClient.Receive(ref RemoteIpEndPoint);
                    string message = Encoding.ASCII.GetString(receiveBytes);
                    MessageHandler(message);
                    Debug.Log("test2");
                    Debug.Log(message);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.Log(ex);
        }
    }

    public void Send(string message)
    {
        try
        {
            Debug.Log("sendsend");
            using (var udpClient = new UdpClient(myPort))
            {
                udpClient.Connect(HostIP, HostPort);
                Byte[] sendBytes = Encoding.ASCII.GetBytes(message);
                udpClient.Send(sendBytes, sendBytes.Length);
            }
        }
        catch (Exception ex)
        {
            Debug.Log(ex);
        }
    }
}
