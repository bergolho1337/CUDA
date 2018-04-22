# CUDA

Repository with	sample codes using the CUDA/C++	library.

### Pre-requisite
1. Install and configure NVIDIA drivers
2. Install and configure the CUDA library

### Installing NVIDIA driver on Fedora 27

To install the NVIDIA proprietary drivers on Fedora 27 you must first update your system:

```bash
$ sudo dnf update
```
Then install the lastest NVIDIA drivers from the DNF repository.

```bash
$ sudo dnf install xorg-x11-drv-nvidia akmod-nvidia
$ sudo dnf install xorg-x11-drv-nvidia-cuda
```
### Installing CUDA libraries on Fedora 27

Download the last version of the CUDA Toolkit runfile. For me it was the Cuda Version v9.1 **Linux-x86_64-Fedora-25-v9.1.run**

`CUDA Toolkits` = https://developer.nvidia.com/cuda-downloads

The latest NVIDIA compiler (NVCC) only works with a version of GNU/GCC compiler that is previous than version 7. So you need to install GCC 6.3.0 for instance, which RPM package could be obtain here:

`GNU/GCC Compiler v6.3.0` = https://drive.google.com/file/d/1t4WrgvpEP-6_NN3qMJhz9MS3CJhHrHKc/view

`GNU/GCC Compiler v5.3.0` = https://drive.google.com/file/d/0B7S255p3kFXNbTBneHgwSzBodFE/view

`GNU/GCC Compiler v4.9.0` = https://drive.google.com/file/d/1Pwq1ua80dGM72i7rpDNAIIdfcR1WK-hG/view

After downloading it make a symbolic link between this version of GCC and the version NVCC uses by using these commands:

```bash
$ sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc 
$ sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```

The GNU/GCC version for each CUDA release can be checked below:

- **CUDA 4.1 release** 
	- gcc 4.5 is now supported. gcc 4.6 and 4.7 are unsupported.
- **CUDA 5.0 release** 
	- gcc 4.6 is now supported. gcc 4.7 is unsupported.
- **CUDA 6.0 release** 
	- gcc 4.7 is now supported.
- **CUDA 7.0 release**
	- gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
- **CUDA 7.5 release**
	- gcc 4.8 is fully supported, with 4.9 support on Ubuntu 14.04 and Fedora 21.
- **CUDA 8 release**
	- gcc 5.3 is fully supported on Ubuntu 16.06 and Fedora 23.
- **CUDA 9 release**
	- gcc 6 is fully supported on Ubuntu 16.04, Ubuntu 17.04 and Fedora 25.

The next step is to solve some bugs with the `floatn.h` header file. There is an problem when we define large float types. To prevent this, we need to add the following lines to the header file (After line 37):

```bash
		+ #if CUDART_VERSION
		+ #undef __HAVE_FLOAT128 0
		+ #define __HAVE_FLOAT128 0
		+ #endif
		
```
After this we proceed normally installing the remaning CUDA-drivers.

```bash
$ sudo dnf install nvidia-driver akmod-nvidia kernel-devel nvidia-driver-libs.i686 vulkan.i686 cuda nvidia-driver-cuda cuda-devel nvidia-driver-NVML-devel
```
Finally execute the runfile that we downloaded, specifing a folder to store some temporary files during the setup.

```bash
$ sudo sh cuda_9.1.85_387.26_linux.run --tmpdir=/home/username/tmp/ --override
```

A minor issue that you might encounter is related to errors on the `host_config.h` file, which should be located on `/usr/local/cuda-9.1/include/crt`. The errors might be related to the version of GCC, so I just comment with a simple `//` the error line.

```bash
#if __GNUC__ > 6

//#error -- unsupported GNU version! gcc versions later than 6 are not supported!

#endif /* __GNUC__ > 6 */ 
```
