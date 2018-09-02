#define ARRAY_DIM 784 // 28*28

// image is a 28x28xN array (N images) of bytes (each pixel is 8 bit grayscale)
// weights is a 28x28x10 array, holding the weights for each digit, 0 - 9
__attribute__((reqd_work_group_size(10000,1,1)))
__kernel void linear_classifier(global const unsigned char * restrict images, 
								constant short * restrict weights,
								global unsigned char * restrict guesses)
{
	int image_array_index = get_global_id(0)*ARRAY_DIM; 
	unsigned char guess = 0;
	int score[10] = {0};
	
	// NOTE: unroll factor depends on size of the target FPGA
	#pragma unroll 8
	for (int x = 0; x < ARRAY_DIM; x++){
		#pragma unroll
		for (int i = 0; i < 10; i++){
			score[i] += images[image_array_index+x]*weights[i*ARRAY_DIM+x];
		}
	}

	// Determine highest score
	#pragma unroll
	for (unsigned char i = 1; i < 10; i++)
		if (score[i] > score[guess]) guess = i;
	guesses[get_global_id(0)] = guess;
}
