# Hailo 4.21 部署包

这三个文件是 vision-hailo 镜像构建的输入，登录 [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) 下载（在 Software Downloads → Archive 区翻 4.21）。

## 文件清单

| 文件 | 用途 | 装在哪 |
|---|---|---|
| `hailort_4.21.0_arm64.deb` | HailoRT C 用户态库 | **镜像里** (Dockerfile `dpkg -i`) |
| `hailort-4.21.0-cp311-cp311-linux_aarch64.whl` | Python 3.11 绑定 | **镜像里** (Dockerfile `pip install`) |
| `hailort-pcie-driver_4.21.0_all.deb` | PCIe 内核驱动 (DKMS) | **宿主机** (备选；推荐用 Frigate 源码方式装) |

## 宿主机装内核驱动 (Bookworm, 一次性)

**推荐 Frigate 源码方式**（最可控）：

```bash
sudo apt update && sudo apt install -y build-essential cmake git wget linux-headers-$(uname -r) dkms
git clone --depth 1 --branch v4.21.0 https://github.com/hailo-ai/hailort-drivers.git
cd hailort-drivers/linux/pcie && sudo make all && sudo make install
cd ../../ && ./download_firmware.sh
sudo mkdir -p /lib/firmware/hailo
sudo mv hailo8_fw.*.bin /lib/firmware/hailo/hailo8_fw.bin
sudo cp ./linux/pcie/51-hailo-udev.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe hailo_pci
sudo reboot
```

或者直接用 .deb：`sudo dpkg -i hailort-pcie-driver_4.21.0_all.deb`（需要先装 DKMS + linux-headers）。

## 验证

```bash
ls /dev/hailo0                          # 设备节点存在
modinfo -F version hailo_pci            # 4.21.0
```
