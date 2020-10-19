#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "mpi.h"
#include "filters.h"

int type, width, height, start, stop, result_row, tmp_rank;

int get_number(FILE* in) {
	char nr[10];
    int nr_len = 0;
    //set t to a value that can't be found in the file at that pos
    char t = 'a';
    while(t != ' ' && t != '\n') {
    	fread(&t, 1, 1, in);
    	nr[nr_len] = t;
    	nr_len++;

    }
    nr[nr_len] = '\0';
    
    int val = 0;
    int idx = 0;
    for(int i = nr_len - 2; i >= 0; i--) {
    	int aux = (nr[idx] - '0');
    	val += pow(10,i)*aux;
    	idx++;
    }
    return val;

}

void read_img(FILE* in, int width, int height, int type, unsigned char* image) {
	
    for(int i = 0; i < height*width; i++) {
    	fread(&image[i], 1, 1, in);
    }
}

void write_header(FILE* out, int width, int height, int maxval, int type) {

	if(type == 1) {
    	fwrite("P5", 1, 2, out);
    } else {
    	fwrite("P6", 1, 2, out);
    }
    char c = '\n';
    fwrite(&c, 1, 1, out);

    char s[30];
    memset(s, 0, 30);
   	sprintf(s, "%d", width/type);
   	s[strlen(s)] = ' ';
   	sprintf(s + strlen(s), "%d", (height));
    fwrite(s, 1, strlen(s), out);
    fwrite(&c, 1, 1, out);

    char maxim[4];
    sprintf(maxim, "%d", maxval);
    fwrite(maxim, 1, 3, out);
    fwrite(&c, 1, 1, out);
}

void apply_filter_on_pixel(float filter[3][3], unsigned char* image, int row, int col, unsigned char* result) {

	int filter_row = 2;
	int filter_col = 2;
	int img_row = row - 1;
	float sum = 0.0;
	while(filter_row >= 0 && img_row <= row + 1) {

		if(img_row >= 0 && img_row < height) {
			filter_col = 2;
			for(int j = col -type; j <= col+type; j+=type) {
				if(j >= 0 && j < width) {
					sum += image[img_row * width + j] * filter[filter_row][filter_col];
				}

				filter_col--;
			}
		}

		img_row++;
		filter_row--;

	}
	
	if(sum < 1) {
		result[(row - start)*width + col] = 0;
		return;
	} 

	if(sum > 255) {
		result[(row - start)*width + col] = 255;
		return;	
	}
	
	unsigned char res = (unsigned char)sum;
	result[(row - start)*width + col] = res;	
}

void apply_filter(float filter[3][3], unsigned char* image, int start, int stop, unsigned char* result) {
	for(int i = start; i <= stop; i++) {
		for(int j = 0; j < width; j++) {
			apply_filter_on_pixel(filter, image, i, j, result);
		}
	}
}


void determine_and_apply_filter(char name[30], unsigned char* image, int start, int stop, unsigned char* result) {
		if(strcmp(name, "smooth") == 0){
			apply_filter(smoothing, image, start, stop, result);
		}
		if(strcmp(name, "blur") == 0) {
			apply_filter(gaussian_blur, image, start, stop, result);
		}
		if(strcmp(name,"mean") == 0) {
			apply_filter(mean_removal, image, start, stop, result);
		}
		if(strcmp(name,"sharpen") == 0) {
			apply_filter(sharpen, image, start, stop, result);
		}
		if(strcmp(name,"emboss") == 0) {
			apply_filter(emboss, image, start, stop, result);
		}
}


int main (int argc, char *argv[]) {
	int   numtasks, rank, len;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    tmp_rank = rank;
    if(rank == 0) {
		char img_name[strlen(argv[1])];
		strcpy(img_name, argv[1]);


		FILE *in=fopen(img_name,"rb");
	    
	    //read type
	    fseek(in, 1, SEEK_SET);
	    char t;
	    fread(&t, 1, 1, in);
	    type = t - '0';

	    //ignore the comment
	    fseek(in, 47, SEEK_CUR);

	    //read height and width
	    width = get_number(in);
	    height = get_number(in);
	    int maxval = get_number(in);

	    if(type == 5) {
	    	type = 1;
	    } else {
	    	type = 3;
	    }
	    
	    width = type * width;
	    unsigned char* image = calloc(height*width, sizeof(unsigned char));

	    //read the pixels
	    read_img(in, width, height, type, image);
	    

	    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(image, width*height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	   
	    fclose(in);
	 	
	    start = rank * ceil(((height-1) * 1.0)/numtasks);
		stop = (rank + 1) * ceil(((height-1) * 1.0)/numtasks) < (height-1) ? (rank + 1) * ceil(((height-1) * 1.0)/numtasks) : (height-1);
		
		unsigned char* result = (unsigned char*)calloc((stop-start + 1)*width, sizeof(unsigned char));
		determine_and_apply_filter(argv[2], image, start, stop, result);
		for(int i = start; i <= stop; i++) {
			for(int j = 0; j < width; j++) {
				image[i*width + j] = result[(i - start)*width + j];
			}
		}
		
		//in case of multiple filters
		if(argc > 3) {
			if(rank + 1 <= numtasks - 1) {
				MPI_Send(&result[(stop - start)*width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
			}

			unsigned char* next_line = calloc(width, sizeof(unsigned char));
			if(rank + 1 <= numtasks - 1) {
				MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				for(int k = 0; k < width; k++) {
					image[(stop+1)* width + k] = next_line[k];
				}
			}

			for(int i = 3; i < argc; i++) {
				
				determine_and_apply_filter(argv[i], image, start, stop, result);
				for(int i = start; i <= stop; i++) {
					for(int j = 0; j < width; j++) {
						image[i*width + j] = result[(i - start)*width  + j];
					}
				}

				if(rank + 1 <= numtasks - 1) {
					MPI_Send(&result[(stop - start)*width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
				}

				if(rank + 1 <= numtasks - 1) {
					MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					for(int k = 0; k < width; k++) {
						image[(stop+1)*width + k] = next_line[k];
					}
				}
			}

			free(next_line);

		}

		FILE *out;
		char file_name[30];
		memset(file_name, 0, 30);
		sprintf(file_name, "%s", "out_");

		if(argc <= 3) {
			strcat(file_name, argv[2]);
		}
		if(type == 1) {
			strcat(file_name, ".pgm");
			out=fopen(file_name,"wb");
		} else {
			strcat(file_name, ".pnm");
			out=fopen(file_name,"wb");
		}

		/*start writing the image
		  only process 0 does this
		*/
	    write_header(out, width, height, maxval, type);

	    for(int i = 0; i < (stop-start) + 1; i++) {
	    	for(int j = 0; j < width; j++) {
	    		fwrite(&result[i*width + j], 1, 1, out);
	    	}
    	}
	    	 
	    free(result);

	    int prev_end = stop;
	    for(int i = 1; i < numtasks; i++) {
	    	int begin = i * ceil(((height-1) * 1.0)/numtasks);
			int end = (i + 1) * ceil(((height-1) * 1.0)/numtasks) < (height-1) ? (i + 1) * ceil(((height-1) * 1.0)/numtasks) : (height-1);

			if(prev_end == begin) {
				begin++;
				prev_end = end;
			}
	    	unsigned char* res = calloc((end-begin + 1)*width, sizeof(unsigned char));
			MPI_Recv(res, (end-begin + 1) * width, MPI_UNSIGNED_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
			for(int idx = 0; idx < end-begin + 1; idx++) {
		    	for(int j = 0; j < width; j++) {
		    		fwrite(&res[idx*width + j], 1, 1, out);
		    	}
    	   	}
 
	    	free(res);
		}

	    fclose(out);
	    free(image);


	} else {

	 	MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    unsigned char*  image = calloc(height*width, sizeof(unsigned char));
	    MPI_Bcast(image, width*height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	    
		start = rank * ceil(((height-1) * 1.0)/numtasks);
		stop = (rank + 1) * ceil(((height-1) * 1.0)/numtasks) < (height-1) ? (rank + 1) * ceil(((height-1) * 1.0)/numtasks) : (height-1);

		//make sure 2 processes don't compute the same line
		int prev_end = (rank) * ceil(((height-1) * 1.0)/numtasks) < (height-1) ? (rank) * ceil(((height-1) * 1.0)/numtasks) : (height-1);
		if(prev_end == start) {
			start++;
		}

		
		unsigned char* result = (unsigned char*)calloc((stop-start + 1) * width,sizeof(unsigned char));
		determine_and_apply_filter(argv[2], image, start, stop, result);

		for(int i = start; i <= stop; i++) {
			for(int j = 0; j < width; j++) {
				image[i*width + j] = result[(i - start)*width  + j];
			}
		}

		//multiple filters
		if(argc > 3) {
			if(rank + 1 <= numtasks - 1) {
				MPI_Send(&result[(stop - start)*width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
			}

			unsigned char* next_line = calloc(width, sizeof(unsigned char));
			MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
			for(int k = 0; k < width; k++) {
					image[(start-1)*width + k] = next_line[k];
			}

			MPI_Send(&result[0], width, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD);
		
			if(rank + 1 <= numtasks - 1) {
				MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for(int k = 0; k < width; k++) {
					image[(stop+1)*width + k] = next_line[k];
				}
			}

			for(int i = 3; i < argc; i++) {
				
				determine_and_apply_filter(argv[i], image, start, stop, result);
				for(int idx = start; idx <= stop; idx++) {
					for(int j = 0; j < width; j++) {
						image[width*idx + j] = result[(idx - start)*width + j];
					}
				}
				
				if(rank + 1 <= numtasks - 1) {
					MPI_Send(&result[(stop - start)*width], width, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD);
				}

				MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for(int k = 0; k < width; k++) {
						image[(start-1)*width + k] = next_line[k];
				}

				MPI_Send(&result[0], width, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD);

				if(rank + 1 <= numtasks - 1) {
					MPI_Recv(next_line, width, MPI_UNSIGNED_CHAR, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					for(int k = 0; k < width; k++) {
						image[(stop+1)*width + k] = next_line[k];
					}
				}	
			}

			free(next_line);
		}

		MPI_Send(result, (stop - start + 1)*width, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
	    	 
	    free(result);
	    free(image);
		
	}

    MPI_Finalize();

}