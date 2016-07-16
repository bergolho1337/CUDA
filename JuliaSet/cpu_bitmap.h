/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"
#include <stdio.h>
#include <stdlib.h>

struct CPUBitmap {
    unsigned char    *pixels;
    int     x, y;
    void    *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL )
    {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    CPUBitmap (FILE *file, void *d = NULL)
    {
      int i, val;
      unsigned char c;
      char line[105];
      fgets(line,100,file);
      fgets(line,100,file);
      fscanf(file,"%s",line);
      // Le as dimensoes da imagem
      x = atoi(line);
      fscanf(file,"%s",line);
      y = atoi(line);
      // Aloca memoria para o pixels
      pixels = new unsigned char[x*y*4];
      // Copiar os pixels
      i = 0;
      while (fscanf(file,"%d",&val) != EOF)
      {
        c = (unsigned char)val;
        pixels[i] = c;
        i++;
      }
      dataBlock = d;
    }

    ~CPUBitmap()
    {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return x * y * 4; }

    void display_and_exit( void(*e)(void*) = NULL )
    {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        bitmapExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
        glutInitWindowSize( x, y );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

    // Send the data of the image to a .txt file to be read in other computer
    void toTxt ()
    {
      int i;
      FILE *file = fopen("julia.txt","w+");
      fprintf(file,"P3\n");
      fprintf(file,"# julia.ppm\n");
      fprintf(file,"%d %d\n",x,y);
      fprintf(file,"255\n");
      for (i = 0; i < image_size(); i += 4)
      {
        fprintf(file,"%u %u %u %u\n",(int)pixels[i],(int)pixels[i+1],(int)pixels[i+2],(int)pixels[i+3]);
      }
      fclose(file);
    }

     // static method used for glut callbacks
    static CPUBitmap** get_bitmap_ptr( void )
    {
        static CPUBitmap   *gBitmap;
        return &gBitmap;
    }

   // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y)
    {
        switch (key) {
            case 27:
                CPUBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
                    bitmap->bitmapExit( bitmap->dataBlock );
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void )
    {
        CPUBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glFlush();
    }
};

#endif  // __CPU_BITMAP_H__
