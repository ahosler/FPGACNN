#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
// #include <time.h>
// #include <sys/time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../../shared/defines.h"
#include "../../shared/utils.h"
//#include "/usr/local/intelFPGA/17.1/hld/include/kernel_headers/ihc_apint.h"
//#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_precision_integers : enable

using namespace aocl_utils;

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;

cl_uchar *input_images = NULL, *output_guesses = NULL, *reference_guesses = NULL;
// How do they want us to do this?
//cl_float *input_weights = NULL;
void *input_weights = NULL;
//cl_short *input_weights_16 = NULL;
//cl_char *input_weights_8 = NULL;
cl_char *input_weights_expanded = NULL;
cl_mem input_images_buffer, input_weights_buffer, output_guesses_buffer;

// Global variables.
std::string imagesFilename;
std::string labelsFilename;
std::string aocxFilename;
std::string deviceInfo;
unsigned int n_items;
bool use_fixed_point;
bool use_single_workitem;
int width;			// Global to hold bit width

//#define ap_int<4> int4_t;
//ap_int<4> int4_t width4bit;

// Function prototypes.
void classify();
void initCL();
void cleanup();
void teardown(int exit_status = 1);
void print_usage();

int main(int argc, char **argv) {
	// Parsing command line arguments.
	Options options(argc, argv);
	
	// If you want to parameterize this file, add code here to parse command line options	
	if(options.has("width")){
		width = options.get<int>("width");
	} else {
		width = 32;
	}

	if(options.has("help")){
		print_usage();
		return 0;
	}
	
	if(options.has("images")) {
		imagesFilename = options.get<std::string>("images");
	} else {
		imagesFilename = "t10k-images.idx3-ubyte";
		printf("Defaulting to images file \"%s\"\n", imagesFilename.c_str());
	}
	
	if(options.has("labels")) {
		labelsFilename = options.get<std::string>("labels");  
	} else {
		labelsFilename = "t10k-labels.idx1-ubyte";
		printf("Defaulting to labels file \"%s\"\n", labelsFilename.c_str());
	}
	
	if(options.has("fixed_point")) {
		use_fixed_point = true; 
	} else {
		use_fixed_point = false;
	}
	
	if(options.has("single_workitem")) {
		use_single_workitem = true; 
	} else {
		use_single_workitem = false;
	}
	
	// Relative path to aocx filename option.
	if(options.has("aocx")) {
		aocxFilename = options.get<std::string>("aocx");  
	} else {
		aocxFilename = "linear_classifier_fp";
		printf("Defaulting to aocx file \"%s\"\n", aocxFilename.c_str());
	}
	
	// Read in the images and labels
	n_items = parse_MNIST_images(imagesFilename.c_str(), &input_images);
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		return -1;
	}
	if (n_items != parse_MNIST_labels(labelsFilename.c_str(), &reference_guesses)){
		printf("ERROR: Number of labels does not match number of images\n");
		return -1;
	}
	
	// Initializing OpenCL and the kernels.
	// Make sure you allocate the right size buffer for your weights
	// TODO: Check the 4 implemention
	output_guesses = (cl_uchar*)alignedMalloc(sizeof(cl_uchar) * n_items);
	if(width == 16) {
		input_weights = (cl_short*)alignedMalloc(sizeof(cl_short) * FEATURE_COUNT * NUM_DIGITS);
	} else if (width == 8) {
		input_weights = (cl_char*)alignedMalloc(sizeof(cl_char) * FEATURE_COUNT * NUM_DIGITS);
	} else if (width == 4) {
		input_weights = (cl_char*)alignedMalloc(sizeof(cl_char) * FEATURE_COUNT * NUM_DIGITS/2);
	} else {
		input_weights = (cl_float*)alignedMalloc(sizeof(cl_float) * FEATURE_COUNT * NUM_DIGITS);
	}
	// Read in the weights from the weights files
	// Make sure you read from the right file.
	for (unsigned i = 0; i < NUM_DIGITS; i++){
		char weights_file[256];
		if (use_fixed_point)
			snprintf(weights_file, 256, "weights_fxp%d/weights_%d_fxp%d", width, i, width);
		else
			snprintf(weights_file, 256, "weights_fp/weights_%d", i);
		if (!read_weights_file(weights_file, (float*)((char*)input_weights+(FEATURE_COUNT*i*width)/8), width)){
			printf("ERROR: Failed to read in weights\n");
			return -1;
		}
	}
	//Expand input weights
	input_weights_expanded = (cl_char*)alignedMalloc(sizeof(cl_char) * FEATURE_COUNT * NUM_DIGITS);
	for (int i = 0; i < FEATURE_COUNT * NUM_DIGITS / 2; i++) {
		//printf("%d", ((char*)input_weights)[i]);
		input_weights_expanded[2*i] = (((char*)input_weights)[i] & 240) >> 4;
		input_weights_expanded[(2*i)+1] = ((char*)input_weights)[i] & 15;
		//printf("%d", ((char*)input_weights_expanded)[i]);
		//printf("%d", ((char*)input_weights_exapnded)[i+1]);
	}

		
	initCL();

	// Start measuring time
	double start = get_wall_time();
	// Call the classifier.
	classify();
	
	// Stop measuring time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
   
	int correct = 0;
	for (unsigned i = 0; i < n_items; i++){
		if (output_guesses[i] == reference_guesses[i]) correct++;
	}
	printf("Classifier accuracy: %.2f%%\n", (float)correct*100/n_items);
	
	// Teardown OpenCL.
	teardown(0);
}

void classify() {
	size_t size = 1;
	cl_int status;
	cl_event event;
	const size_t global_work_size = n_items;
	
	// Create kernel input and output buffers.
	// Make sure you create the right size buffer
	input_images_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * FEATURE_COUNT * n_items, NULL, &status);
	checkError(status, "Error: could not create input image buffer");
	// Does width work?
	if (width != 4) {
		input_weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (width * FEATURE_COUNT * NUM_DIGITS) / 8, NULL, &status);
	} else {
		input_weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (width * FEATURE_COUNT * NUM_DIGITS) / 4, NULL, &status);
	}
	checkError(status, "Error: could not create input image buffer");
	output_guesses_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * n_items, NULL, &status);
	checkError(status, "Error: could not create output guesses buffer");
	
	// Copy data to kernel input buffer.
	// TODO: Make sure you pass the right input weights (and the right size)
	status = clEnqueueWriteBuffer(queue, input_images_buffer, CL_TRUE, 0, sizeof(unsigned char) * FEATURE_COUNT * n_items, input_images, 0, NULL, NULL);
	checkError(status, "Error: could not copy data into device");
	// Does width work?
	if (width != 4) {
		status = clEnqueueWriteBuffer(queue, input_weights_buffer, CL_TRUE, 0, (width * FEATURE_COUNT * NUM_DIGITS) / 8, input_weights, 0, NULL, NULL);
	} else {
		status = clEnqueueWriteBuffer(queue, input_weights_buffer, CL_TRUE, 0, (width * FEATURE_COUNT * NUM_DIGITS) / 4, input_weights_expanded, 0, NULL, NULL);
	}
	checkError(status, "Error: could not copy data into device");
	
	// Set the arguments for data_in, data_out and sobel kernels.
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_images_buffer);
	checkError(status, "Error: could not set argument 0");
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&input_weights_buffer);
	checkError(status, "Error: could not set argument 1");
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_guesses_buffer);
	checkError(status, "Error: could not set argument 2");
	
	int n_items = 10000;
	if (use_single_workitem){
		status = clSetKernelArg(kernel, 3, sizeof(int), &global_work_size);
		checkError(status, "Error: could not set argument 2");
	}
	
	// Enqueue the kernel. //
	if (use_single_workitem){
		status = clEnqueueTask(queue, kernel, 0, NULL, &event);
	} else {
		status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &event);
	}
	checkError(status, "Error: failed to launch data_in");
	
	// Wait for command queue to complete pending events.
	status = clFinish(queue);
	checkError(status, "Kernel failed to finish");

	clReleaseEvent(event);
	
	// Read output buffer from kernel.
	status = clEnqueueReadBuffer(queue, output_guesses_buffer, CL_TRUE, 0, sizeof(unsigned char) * n_items, output_guesses, 0, NULL, NULL);
	checkError(status, "Error: could not copy data from device");
}

void initCL() {
	cl_int status;

	// Start everything at NULL to help identify errors.
	kernel = NULL;
	queue = NULL;
	
	// Locate files via. relative paths.
	if(!setCwdToExeDir()) {
		teardown();
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA");
	if (platform == NULL) {
		teardown();
	}

	// Get the first device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError (status, "Error: could not query devices");

	char info[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
	deviceInfo = info;

	// Create the context.
	context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Error: could not create OpenCL context");

	// Create the command queues for the kernels.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
	std::cout << "Using AOCX: " << binary_file << "\n";
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 1, &device, "", NULL, NULL);
	checkError(status, "Error: could not build program");
	
	// Create the kernel - name passed in here must match kernel name in the original CL file.
	kernel = clCreateKernel(program, "linear_classifier", &status);
	checkError(status, "Failed to create kernel");
}

void cleanup() {
	// Called from aocl_utils::check_error, so there's an error.
	teardown(-1);
}

void teardown(int exit_status) {
	if(kernel) clReleaseKernel(kernel);
	if(queue) clReleaseCommandQueue(queue);
	if(input_images) alignedFree(input_images);
	if(input_weights) alignedFree(input_weights);
	if(input_weights_expanded) alignedFree(input_weights_expanded);
	if(reference_guesses) alignedFree(reference_guesses);
	if(output_guesses) alignedFree(output_guesses);
	if(input_images_buffer) clReleaseMemObject(input_images_buffer);
	if(output_guesses_buffer) clReleaseMemObject(output_guesses_buffer);
	if(program) clReleaseProgram(program);
	if(context) clReleaseContext(context);
	
	exit(exit_status);
}

void print_usage() {
	printf("\nUsage:\n");
	printf("\tlinear_classifier  [--images=<MNIST images file>] [--labels=<MNIST labels file>] \n\n");
	printf("Options:\n\n");
	printf("--images=<MNIST images file>\n");
	printf("\tThe relative path to the MNIST images file.\n");
	printf("--labels=<MNIST labels file>\n");
	printf("\tThe relative path to the MNIST labels file.\n");
	printf("--aocx=<AOCX file>\n");
	printf("\tThe relative path to the .aocx file to use.\n");
	printf("--fixed_point\n");
	printf("\tUse fixed point weights.\n");
	printf("--single_workitem\n");
	printf("\tUse a single workitem kernel.\n");
}
