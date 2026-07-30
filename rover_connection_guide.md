# Waveshare UGV Rover PT – Remote Access Guide
**Team Crater | Tulane University**

---

## Overview

The rover runs Ubuntu 22.04 on a Jetson Orin Nano. It connects to the **Tulane IoT** WiFi network on boot. To access it remotely from any network (eduroam, home, hotspot, etc.), we use **Tailscale**, a free VPN that connects all your devices into a private network.

- **Rover Tailscale IP:** `100.68.244.40` *(permanent, never changes)*
- **Rover USB IP:** `192.168.55.1` *(only works when physically plugged in via USB)*
- **Rover Tulane IoT IP:** `10.130.37.69` *(only works when you are also on Tulane IoT)*
- **SSH Username:** `jetson`
- **SSH Password:** `jetson`

---

## Method 1: SSH via Tailscale (Recommended — works from anywhere)

This is the primary method. It works from eduroam, home WiFi, hotspots, or any network.

### Step 1: Install Tailscale on your device

**Option A — Download from website:**
Go to [https://tailscale.com/download](https://tailscale.com/download) and download the installer for your OS (Mac, Windows, Linux).

**Option B — Mac (Homebrew):**
```bash
brew install tailscale
```

**Option C — Linux:**
```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

**Option D — Windows:**
Download the `.exe` installer from [https://tailscale.com/download/windows](https://tailscale.com/download/windows)

---

### Step 2: Sign in to the Team Crater Tailscale account

When prompted to sign in, use:
- **Email:** `teamcratertulane@gmail.com`
- **Password:** `Team@Crater2025`

> ⚠️ Make sure you sign in with this account, not your personal one. All team members must be on the same Tailscale network to reach the rover.

---

### Step 3: SSH into the rover

Once Tailscale is running and connected, open a terminal and run:

```bash
ssh jetson@100.68.244.40
```

Enter password: `jetson`

You're in! The rover must be powered on and connected to Tulane IoT for this to work.

---

## Method 2: SSH via USB Cable (Fallback — works without WiFi)

Use this if Tailscale isn't working or the rover isn't connected to any WiFi. Requires a **Micro USB data cable** (not just a charging cable).

1. Plug the Micro USB port of the Jetson Orin Nano into your computer
2. Open a terminal and run:

```bash
ssh jetson@192.168.55.1
```

Enter password: `jetson`

> This IP is fixed and hardwired — it always works over USB regardless of network state.

---
## Method 3: SSH via Access Popup
Sometimes, the rover will not connect to Tulane IoT in a timely manner, or sometimes at all, upon rebooting after being shut off for awhile. In this case, it is possible to connect to a hotspot that Jetson automatically boots upon startup named, "Access Popup".

### Step 1: Connect to Access Popup network. 

Find Access Popup among the available wifi networks. The password is: 1234567890

### Step 2: Open the terminal and run: ssh jetson@10.42.0.1
Enter the passcode: jetson

### Step 3: Run: sudo nmcli connection up "Tulane IoT"
*Note that this is for use on the Tulane Campus. You will have to connect to another wifi if elsewhere.

### Step 4: SSH into the rover

Once Tailscale is running and connected, open a terminal and run:

```bash
ssh jetson@100.68.244.40
```

Enter password: `jetson`

You're in! 

---

## Method 4: SSH via Tulane IoT (Same network only)

If your device is also connected to the **Tulane IoT** network, you can connect directly:

```bash
ssh jetson@10.130.37.69
```

> ⚠️ This only works if both your device and the rover are on Tulane IoT. It does **not** work from eduroam.

---

## Accessing the Web Interface

The rover has a browser-based control panel for camera view and manual control.

| Interface | URL |
|---|---|
| **Main control panel** | `http://100.68.244.40:5000` |
| **JupyterLab** | `http://100.68.244.40:8888/lab` |

Open these in any browser while connected to Tailscale. JupyterLab password is `jetson`.

---

## Checking the Rover's Connection Status

SSH in and run:

```bash
nmcli device status      # shows what network it's connected to
ip addr show wlan0       # shows current WiFi IP
tailscale ip             # shows Tailscale IP (should be 100.68.244.40)
```

The OLED screen on the rover also displays:
- `E` line: Ethernet IP
- `W` line: WiFi IP (Tulane IoT or hotspot)

---

## Downloading Files from the Rover

To copy files to your computer, run this from your **local terminal** (not inside SSH):

```bash
# Download entire home directory (skipping large compiled files)
rsync -avz --exclude='ugv_jetson/ugv-env' --exclude='opencv_cuda' \
  -e ssh jetson@100.68.244.40:/home/jetson/ ~/Desktop/rover_files

# Download a specific folder
scp -r jetson@100.68.244.40:/home/jetson/ugv_jetson ~/Desktop/ugv_jetson
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection timed out` on Tailscale IP | Rover may still be booting — wait 60s and retry |
| `Connection timed out` on USB IP | Check USB cable is a data cable, not charge-only |
| Web panel at `:5000` gives Internal Server Error | SSH in and run `sudo reboot`, then wait 60s |
| Tailscale shows rover as offline | Rover is off, or not connected to any internet |
| Can't reach rover from eduroam | Use Tailscale (Method 1) — eduroam blocks direct LAN access |

---

## Rebooting the Rover Remotely

If something is stuck (web panel down, app crashed, etc.):

```bash
sudo reboot
```

Wait about 60 seconds, then reconnect.

---

## Quick Reference

```
Tailscale IP:     100.68.244.40
USB IP:           192.168.55.1
SSH user:         jetson
SSH password:     jetson
Web panel:        http://100.68.244.40:5000
JupyterLab:       http://100.68.244.40:8888/lab
Tailscale login:  teamcratertulane@gmail.com / Team@Crater2025
```
