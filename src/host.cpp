#include "xcl2.hpp"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <cmath>
#include <vector>
//#include <ap_cint.h>


//void deflate(uint512 *in_buf, int in_size,
//                uint512 *out_buf, int *out_size);


int deflate_test(std::string& binary, std::string& inFilePath){

    std::string outFilePath = inFilePath + ".zlib";

    std::ifstream inFile(inFilePath.c_str(), std::ifstream::binary);
    if (!inFile) {
        std::cout << "Unable to open input file" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ofstream outFile(outFilePath.c_str(), std::ofstream::binary);
    if (!outFile) {
        std::cout << "Unable to open input file" << std::endl;
        exit(EXIT_FAILURE);
    }

    int32_t input_size;
    //int32_t output_size;

    inFile.seekg(0, inFile.end);
    input_size = inFile.tellg();
    inFile.seekg(0, inFile.beg);


    std::vector<uint8_t, aligned_allocator<uint8_t>> in(input_size);
    std::vector<uint8_t, aligned_allocator<uint8_t>> out(input_size*2);
    std::vector<int32_t, aligned_allocator<int32_t>> output_size(1);
    inFile.read((char*)in.data(), input_size);

    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    // Create Program
    auto fileBuf = xcl::read_binary_file(binary);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
//    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel deflate_kernel(program, "deflate", &err));
    
    // Allocate Buffer in Global Memory
    OCL_CHECK(err,
              cl::Buffer buffer_in (context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                input_size,
                                in.data(),
                                &err));

    OCL_CHECK(err,
              cl::Buffer buffer_out (context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  input_size*2,
                                  out.data(),
                                  &err));

    OCL_CHECK(err,
              cl::Buffer buffer_outLen (context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  sizeof(int32_t),
                                  output_size.data(),
                                  &err));               

    printf("|-------------------------+-------------------------|\n"
           "| Kernel                  |    Wall-Clock Time (ns) |\n"
           "|-------------------------+-------------------------|\n");

    
    int nargs = 0;
    OCL_CHECK(err, err = deflate_kernel.setArg(nargs++, buffer_in));
    OCL_CHECK(err, err = deflate_kernel.setArg(nargs++, input_size));
    OCL_CHECK(err, err = deflate_kernel.setArg(nargs++, buffer_out));
    OCL_CHECK(err, err = deflate_kernel.setArg(nargs++, buffer_outLen));


    cl::Event krnlEvent;

    struct timeval startTime, stopTime;
    double exec_time;
    double fpga_throughput;

    uint64_t krnl_start, krnl_end;
    double krnl_exec_time;
    double krnl_throughput;

    gettimeofday(&startTime, NULL);
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in},0/* 0 means from host*/));
    OCL_CHECK(err, err = q.finish());
    //std::cout << "Copy data from host to FPGA is done!" << std::endl;

    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(deflate_kernel, NULL, &krnlEvent));
    clWaitForEvents(1, (const cl_event*) &krnlEvent);
    //std::cout << "Kernel execution is done!" << std::endl;

    // Copy output length back to host local memory first
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outLen},CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    //std::cout << "Copy data out size from FPGA to host is done!" << std::endl;
    if (output_size[0] <= 0) {
      exit(EXIT_FAILURE);
    }
    // Copy output data back to host local memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    //std::cout << "Copy data from FPGA to host is done!" << std::endl;

    gettimeofday(&stopTime, NULL);

    krnlEvent.getProfilingInfo(CL_PROFILING_COMMAND_START, &krnl_start);
    krnlEvent.getProfilingInfo(CL_PROFILING_COMMAND_END, &krnl_end);

    std::cout << "Task execution stopped here" << std::endl;

    // Write compressed data to output
    outFile.write((char*)out.data(), output_size[0]);

    // Close file
    inFile.close();
    outFile.close();

    // Calculate compression throughput & compression ratio
    krnl_exec_time = (krnl_end - krnl_start) / 1000000000.0;
    std::cout << "kernel execution time is " << krnl_exec_time << "s\n";
    krnl_throughput = (input_size / 1024.0 / 1024.0 / 1024.0) / krnl_exec_time; 
    std::cout << "Kernel throughput is " << krnl_throughput << "" << std::endl;

    exec_time = (stopTime.tv_usec - startTime.tv_usec) / 1000000.0 + (stopTime.tv_sec - startTime.tv_sec);
    std::cout << "Execution time is " << exec_time << "s\n";
    fpga_throughput = (input_size / 1024.0 / 1024.0 / 1024.0) / exec_time;
    std::cout << "End-to-end throughput is " << fpga_throughput << "" << std::endl;

    double ratio = (double)1.0 * input_size / output_size[0];
    std::cout << "Compression ratio is " << ratio << std::endl;

    return EXIT_SUCCESS;

}


int main(int argc, char* argv[]){
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> + input file path" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string inFilePath = argv[2];

    int ret;

    ret = deflate_test(binaryFile, inFilePath);
    if (ret != 0) {
      std::cout << "Test failed !!!" << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
