#include <stdio.h>
#include <stdlib.h>
#include "ppm_lib.h"

#define divisionFactor 9
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





__global__ void filterSofter(PPMPixel *img,int* filter ,PPMPixel *destination){
        /*
       int gridCounter;
       int finalRed =0;
       int finalGreen  =0;
       int  finalBlue =0;
       int tmpx;
       int tmpy;
       for(int y=0; y<=img->y; y++){ // for each pixel in the image
        for(int x=0; x<=img->x; x++){ 
             gridCounter=0;
            finalRed = 0;
            finalGreen = 0;
            finalBlue =0;
            tmpx =0;
            tmpy =0;
            for(int y2=-2; y2<=2; y2++){ 
                for(int x2=-2; x2<=2; x2++) {  
                      tmpx=x2;
                       tmpy=y2;
                    if( x+x2 > (img->x)-2){
                        tmpx = -x2;
                    }else if( x+x2 < 0 ){
                        tmpx = -x2;
                    }

                    if (y+y2 > (img->y)-2){
                        tmpy = -y2;                    
                    }else if (y+y2 < 0) {
                       tmpy=-y2;
                    }
                        
                    finalRed += img-[(x+tmpx) + (y+tmpy)*img->x].red * filter[gridCounter];  
                    finalGreen +=  img-[(x+tmpx) + (y+tmpy)*img->x].green * filter[gridCounter]; 
                    finalBlue +=  img-[(x+tmpx) + (y+tmpy)*img->x].blue * filter[gridCounter]; 
                    gridCounter++;
                }}
                

                destination[x+y*img->x].red = finalRed / divisionFactor;
                destination-[x+y*img->x].green = finalGreen / divisionFactor;
               destination[x+y*img->x].blue = finalBlue / divisionFactor;
            }
        }*/

        return;
    
}




int main(){

    PPMImage *image, *imageCopy;
    image = readPPM("imageProject.ppm");
    imageCopy = readPPM("imageProject.ppm");
    

int filter[25] = { 0,  0,   0,   0,   0,
                   0 ,  0 ,  0 ,  0 ,  0 ,
                   1 ,  2 ,  3 ,  2 ,  1 ,
                   0  , 0 ,  0 ,  0 ,  0,
                   0 ,  0 ,  0 ,  0 ,  0 };

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
filterSofter<<<10,25>>>(dev_image,dev_filter,dev_imageCopy);
cudaThreadSynchronize();

printf(">%s\n",cudaGetErrorString (cudaGetLastError ()));
/*
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
*/

/* copier le tableau 'c' depuis le GPU vers le CPU */
HANDLE_ERROR( cudaMemcpy( imageCopy->data, dev_image, imageCopy->x*imageCopy->y * 3*sizeof(char), cudaMemcpyDeviceToHost));


//printf("Temps nécessaire :  %3.1f ms\n", time);

writePPM("imageProjectResult.ppm",imageCopy);

/* liberer la memoire allouee sur le GPU */
cudaFree( dev_image );
cudaFree( dev_imageCopy );
cudaFree( dev_filter );

return 0;
}
