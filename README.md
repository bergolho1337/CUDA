# CUDA

Repository with	sample codes using the CUDA/C++	library.

###Pre-requisite
1. Install and configure NVIDIA drivers
2. Install and configure the CUDA library

###Installing NVIDIA driver on Fedora 27

To install the NVIDIA proprietary drivers on Fedora 27 you must first update your system:

```bash
$ sudo dnf update
```
Then install the lastest NVIDIA drivers from the DNF repository.

```bash
$ sudo dnf install xorg-x11-drv-nvidia akmod-nvidia
$ sudo dnf install xorg-x11-drv-nvidia-cuda
```
###Installing CUDA libraries on Fedora 27

Download the last version of the CUDA Toolkit runfile. For me it was the **Linux-x86_64-Fedora-25-v9.1.run**

`CUDA Toolkits` = https://developer.nvidia.com/cuda-downloads

The NVIDIA compiler (NVCC) only works with a version of GNU/GCC compiler that is previous than version 7. So you need to install GCC 6.3.0 for instance, which RPM package could be obtain here:

`GNU/GCC Compiler v6.3.0` = https://drive.google.com/file/d/1t4WrgvpEP-6_NN3qMJhz9MS3CJhHrHKc/view

After downloading it make a symbolic link between this version of GCC and the version NVCC uses by using these commands:

```bash
$ sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc 
$ sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```

There are some bugs with the `floatn.h` header file when we define large float types. To prevent this, we need to add the following lines to the header file (After line 37):

```bash
		+ #if CUDART_VERSION
		+ #undef __HAVE_FLOAT128 0
		+ #define __HAVE_FLOAT128 0
		+ #endif
		
```
Then we install the remaning CUDA-drivers.

```bash
$ sudo dnf install nvidia-driver akmod-nvidia kernel-devel nvidia-driver-libs.i686 vulkan.i686 cuda nvidia-driver-cuda cuda-devel nvidia-driver-NVML-devel
```
And finally execute the runfile that we downloaded, specifing a folder to store some temporary files during the setup.

```bash
$ sudo sh cuda_9.1.85_387.26_linux.run --tmpdir=/home/username/tmp/ --override
```
