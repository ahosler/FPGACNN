#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <sys/time.h>
#include "defines.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

unsigned int convert_endian_4bytes(unsigned int input){
	
	unsigned char* bytes = (unsigned char*) &input;
	
	return (unsigned int) bytes[3] | (((unsigned int) bytes[2]) << 8) | (((unsigned int) bytes[1]) << 16) | (((unsigned int) bytes[0]) << 24);
}

void write_weights_file(char *filename, float *weights, int num_weights) {
	FILE *f = fopen(filename, "wb");
	if (f == NULL){
		printf("ERROR: could not open %s\n",filename);
		return;
	}
	fwrite(weights, sizeof(float), num_weights, f);
	fclose(f);
}

// TODO: You may need to modify this file to read in files with differently sized weights.
bool read_weights_file(char *filename, float *weights, int width) {
	FILE *f = fopen(filename, "rb");
	if (f == NULL){
		printf("ERROR: could not open %s\n",filename);
		return false;
	}
	//printf("%d\n", FEATURE_COUNT);	
	int read_elements = -1;
	if (width == 16) {
		read_elements = fread(weights, sizeof(short), FEATURE_COUNT, f);
	}else if(width == 8) {
		read_elements = fread(weights, sizeof(char), FEATURE_COUNT, f);
	}else if(width == 4) {
		read_elements = fread(weights, sizeof(char), FEATURE_COUNT/2, f);
	}else {	
		read_elements = fread(weights, sizeof(float), FEATURE_COUNT, f);
		//for (int i = 0; i<FEATURE_COUNT; i++) {
			//printf("%f\n",weights[i]);
		//}
	}
	fclose(f);
	//TODO: Uncomment with fix
	if (read_elements != FEATURE_COUNT){
		if(width == 4 && read_elements != FEATURE_COUNT/2) {
			printf("ERROR: read incorrect number of weights from %s\n", filename);
			return false;
		} else {
			return true;
		}
		printf("ERROR: read incorrect number of weights from %s\n", filename);
		return false;
	}
	return true;
}


int parse_MNIST_images(const char* file, unsigned char** X){
	int n_items = 0, magic_num = 0, n_rows = 0, n_cols = 0, elements_read = 0;
	FILE* f = fopen(file, "rb");
	if (f == NULL){
		printf("ERROR: Could not open %s\n",file);
		return 0;
	}
	
	elements_read = fread(&magic_num, sizeof(int), 1, f );
	
	if (convert_endian_4bytes(magic_num) != MNIST_IMAGE_FILE_MAGIC_NUMBER){
		printf("WARNING: Magic number mismatch in %s.\n", file);
	}
	
	elements_read = fread(&n_items, sizeof(int), 1, f );
	elements_read = fread(&n_rows, sizeof(int), 1, f );
	elements_read = fread(&n_cols, sizeof(int), 1, f );
	
	n_items = convert_endian_4bytes(n_items);
	n_rows = convert_endian_4bytes(n_rows);
	n_cols = convert_endian_4bytes(n_cols);
	
	// n_rows * n_cols should equal FEATURE_COUNT (28*28 = 784)
	if (n_rows*n_cols != FEATURE_COUNT){
		printf("ERROR: Unexpected image size in %s.\n", file);
		return 0;
	}
	
	*X = (cl_uchar*)alignedMalloc(n_items*FEATURE_COUNT*sizeof(unsigned char));
	unsigned char* X_ptr = *X;
	// Read in the pixels. Each pixel is 1 byte. There should be n_items*n_rows*n_cols pixels).
	elements_read = fread(X_ptr, sizeof(unsigned char), n_rows*n_cols*n_items, f);
	if (elements_read != n_rows*n_cols*n_items){
		printf("ERROR: Unexpected file length for %s.\n", file);
		free(*X);
		return 0;
	}
	
	return n_items;
}

int parse_MNIST_labels(const char* file, unsigned char** y){
	int n_items = 0;
	int magic_num = 0;
	int elements_read = 0;
	FILE* f = fopen(file, "rb");
	if (f == NULL){
		printf("ERROR: Could not open %s\n",file);
		return 0;
	}
	
	elements_read = fread(&magic_num, sizeof(int), 1, f );
	
	if (convert_endian_4bytes(magic_num) != MNIST_LABEL_FILE_MAGIC_NUMBER){
		printf("WARNING: Magic number mismatch in %s.\n", file);
	}
	
	elements_read = fread(&n_items, sizeof(int), 1, f );
	n_items = convert_endian_4bytes(n_items);
	*y = (cl_uchar*) alignedMalloc(n_items*sizeof(unsigned char));
	
	// Read in the pixels. Each pixel is 1 byte. There should be n_items*n_rows*n_cols pixels).
	elements_read = fread(*y, sizeof(unsigned char), n_items, f);
	if (elements_read != n_items){
		printf("ERROR: Unexpected file length for %s.\n", file);
		free(*y);
		return 0;
	}
	
	return n_items;
}

void parse_arguments(int argc, char *argv[], int *task, float *alpha, int *iterations, int *n_items_limit) {
	// Setting task based on FIRST argument
	*task = UNKNOWN;
	if(argc >= 2) {
		if(strcmp(argv[1], "train") == 0)
			*task = TRAIN;
		else if(strcmp(argv[1], "test") == 0)
			*task = TEST;
	}
	
	if(*task != UNKNOWN) {
		// Set default values.
		*alpha = DEFAULT_ALPHA;
		*iterations = DEFAULT_ITERATIONS;
		*n_items_limit = DEFAULT_N_ITEMS_LIMIT;
		
		// Read command line arguments.
		for(int i = 2; i < argc - 1; i++) {
			if (strcmp(argv[i], "--alpha") == 0)
				*alpha = atof(argv[i + 1]);
			else if (strcmp(argv[i], "--alpha_int") == 0)
				*alpha = atoi(argv[i + 1]);
			else if (strcmp(argv[i], "--iter") == 0)
				*iterations = atoi(argv[i + 1]);
			else if (strcmp(argv[i], "--nitems") == 0)
				*n_items_limit = atoi(argv[i + 1]);
		}
	}
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec*1000 + (double)time.tv_usec /1000;
}
