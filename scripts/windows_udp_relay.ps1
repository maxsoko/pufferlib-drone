param(
    [string]$ListenAddress = "0.0.0.0",
    [string]$WslHost = "127.0.0.1",
    [int]$MavlinkListenPort = 14540,
    [int]$MavlinkForwardPort = 14540,
    [string]$SimulatorMavlinkHost = "127.0.0.1",
    [int]$SimulatorMavlinkPort = 14560,
    [int]$CameraListenPort = 5600,
    [int]$CameraForwardPort = 5600
)

$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public sealed class Ts002UdpRelay {
    private readonly UdpClient mavListen;
    private readonly UdpClient camListen;
    private readonly UdpClient camWsl;
    private readonly IPEndPoint wslMav;
    private readonly IPEndPoint wslCam;
    private readonly IPEndPoint simMav;
    private volatile IPEndPoint wslMavPeer;
    private long mavIn;
    private long mavOut;
    private long camIn;

    public Ts002UdpRelay(
        string listenAddress,
        string wslHost,
        int mavlinkListenPort,
        int mavlinkForwardPort,
        string simulatorMavlinkHost,
        int simulatorMavlinkPort,
        int cameraListenPort,
        int cameraForwardPort
    ) {
        mavListen = Bind(listenAddress, mavlinkListenPort);
        camListen = Bind(listenAddress, cameraListenPort);
        camWsl = new UdpClient(0);
        wslMav = new IPEndPoint(IPAddress.Parse(wslHost), mavlinkForwardPort);
        wslCam = new IPEndPoint(IPAddress.Parse(wslHost), cameraForwardPort);
        simMav = new IPEndPoint(IPAddress.Parse(simulatorMavlinkHost), simulatorMavlinkPort);
    }

    private static UdpClient Bind(string address, int port) {
        UdpClient client = new UdpClient();
        client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
        client.Client.Bind(new IPEndPoint(IPAddress.Parse(address), port));
        return client;
    }

    public void Run() {
        Start("mav-bridge", MavBridge);
        Start("camera-sim-to-wsl", CameraSimToWsl);
        Console.WriteLine("Relay listening. Press Ctrl+C to stop.");
        while (true) {
            Thread.Sleep(2000);
            string wslPeer = wslMavPeer == null ? "unknown" : wslMavPeer.ToString();
            Console.WriteLine(
                "stats mav_to_wsl=" + Interlocked.Read(ref mavIn) +
                " mav_to_sim=" + Interlocked.Read(ref mavOut) +
                " camera_sim_to_wsl=" + Interlocked.Read(ref camIn) +
                " wsl_mav_peer=" + wslPeer
            );
        }
    }

    private static void Start(string name, ThreadStart fn) {
        Thread thread = new Thread(fn);
        thread.IsBackground = true;
        thread.Name = name;
        thread.Start();
    }

    private void MavBridge() {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (true) {
            byte[] data = mavListen.Receive(ref remote);
            if (remote.Address.Equals(simMav.Address) && remote.Port == simMav.Port) {
                IPEndPoint peer = wslMavPeer == null ? wslMav : wslMavPeer;
                mavListen.Send(data, data.Length, peer);
                Interlocked.Increment(ref mavIn);
            } else {
                wslMavPeer = new IPEndPoint(remote.Address, remote.Port);
                mavListen.Send(data, data.Length, simMav);
                Interlocked.Increment(ref mavOut);
            }
        }
    }

    private void CameraSimToWsl() {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (true) {
            byte[] data = camListen.Receive(ref remote);
            camWsl.Send(data, data.Length, wslCam);
            Interlocked.Increment(ref camIn);
        }
    }
}
"@

Write-Host "Forwarding MAVLink ${ListenAddress}:${MavlinkListenPort} -> ${WslHost}:${MavlinkForwardPort}"
Write-Host "Bridging MAVLink ${ListenAddress}:${MavlinkListenPort} <-> ${SimulatorMavlinkHost}:${SimulatorMavlinkPort}"
Write-Host "Forwarding camera  ${ListenAddress}:${CameraListenPort} -> ${WslHost}:${CameraForwardPort}"

$relay = [Ts002UdpRelay]::new(
    $ListenAddress,
    $WslHost,
    $MavlinkListenPort,
    $MavlinkForwardPort,
    $SimulatorMavlinkHost,
    $SimulatorMavlinkPort,
    $CameraListenPort,
    $CameraForwardPort
)
$relay.Run()
