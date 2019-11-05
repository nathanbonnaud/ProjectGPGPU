#include <stdio.h>
#include <stdlib.h>
#include "ppm_lib.h"

#define divisionFactor 9
#define N 500*1000
static void HandleError( cudaError_t err, const char *file, int line ) {
if (err != cudaSuccess) {
printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
exit( EXIT_FAILURE ); }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define CREATOR "PARALLELISME2OPENMP"



PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, rgb_comp_color;
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    if (fread(img->data, sizeof(PPMPixel)*img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "# Created by %s\n",CREATOR);
    fprintf(fp, "%d %d\n",img->x,img->y);

    fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}




// GPU version 1//
// un bloc par pixel et un bloc par ligne on incrémente//  
__global__ void filterSofter(PPMPixel *img,int *filter ,PPMPixel *destination){
     
         
         int  finalRed =0;
         int  finalGreen  =0;
         int  finalBlue =0;
         int indFiltre = 0;
         
         int tidX =threadIdx.x+ blockIdx.x*blockDim.x;
         int l=tidX/500;          
         int c=tidX%500;
         int ll;
         int cc;

        for(int i=-2;i<=2;i++){
            for(int j=-2;j<=2;j++){
            ll=l+i;
            cc=c+j;

            if(ll<0 ){
                ll=-ll;
            }else if(ll>1000){
              ll=l-i;
            }
            if(cc<0 ){
                cc=-cc;
            } else if (cc>500){
              cc=c-j;
            }

            finalRed += img[(ll)*500+(cc)].red * filter[indFiltre];  
            finalGreen +=  img[(ll)*500+(cc)].green * filter[indFiltre]; 
            finalBlue +=  img[(ll)*500+(cc)].blue * filter[indFiltre]; 
            indFiltre++;
        
            }
        }
        
        
         destination[tidX].red =  finalRed / divisionFactor;
         destination[tidX].green = finalGreen / divisionFactor;
         destination[tidX].blue =  finalBlue / divisionFactor;
        
    
}



int main(){

    PPMImage *image, *imageCopy;
    image = readPPM("imageProject.ppm");
    imageCopy = readPPM("imageProject.ppm");
    

int filter[25] = { 1,   2,   0,   -2,   -1,
                           4 ,  8,   0 ,  -8 ,  -4,
                           6  , 12 , 0 ,  -12  , -6 ,
                           4,   8,   0 ,  -8,    -4,
                           1,   2,   0,   -2,   -1 };


PPMPixel *dev_image;
PPMPixel*dev_imageCopy;
int *dev_filter;

//double time;
//cudaEvent_t start,stop;


HANDLE_ERROR( cudaMalloc( (void**)&dev_image, image->x*image->y *3* sizeof(char) ) );
HANDLE_ERROR( cudaMalloc( (void**)&dev_imageCopy, imageCopy->x*imageCopy->y*3 * sizeof(char) ) );
HANDLE_ERROR( cudaMalloc( (void**)&dev_filter, 25 * sizeof(int) ));


/* copier 'a' et 'b' sur le GPU */

HANDLE_ERROR( cudaMemcpy( dev_image, image->data,image->x*image->y *3* sizeof(char),cudaMemcpyHostToDevice));
HANDLE_ERROR( cudaMemcpy( dev_imageCopy, imageCopy->data, imageCopy->x*imageCopy->y *3* sizeof(char),cudaMemcpyHostToDevice));
HANDLE_ERROR( cudaMemcpy( dev_filter, filter, 25 * sizeof(int),cudaMemcpyHostToDevice));


/*cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);
*/
filterSofter<<<500,1000>>>(dev_image,dev_filter,dev_imageCopy);


printf(">%s\n",cudaGetErrorString (cudaGetLastError ()));
/*
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
*/

/* copier le tableau 'c' depuis le GPU vers le CPU */
cudaMemcpy( imageCopy->data, dev_imageCopy, imageCopy->x*imageCopy->y * 3*sizeof(char), cudaMemcpyDeviceToHost);


//printf("Temps nécessaire :  %3.1f ms\n", time);

writePPM("imageProjectResult.ppm",imageCopy);

/* liberer la memoire allouee sur le GPU */
cudaFree( dev_image );
cudaFree( dev_imageCopy );
cudaFree( dev_filter );

return 0;
}
