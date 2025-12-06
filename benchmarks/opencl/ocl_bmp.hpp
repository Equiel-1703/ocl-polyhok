/*
    This header defines structures and a function to generate a BMP image file.

    Ported by: Henrique Gabriel Rodrigues
    Original code by: Prof. Dr. André Rauber Du Bois
*/

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define _bitsperpixel 32
#define _planes 1
#define _compression 0
#define _xpixelpermeter 0x13B // 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B // 0x130B //2835 , 72 DPI

#pragma pack(push, 1)
typedef struct
{
  uint8_t signature[2];
  uint32_t filesize;
  uint32_t reserved;
  uint32_t fileoffset_to_pixelarray;
} Fileheader;
typedef struct
{
  uint32_t dibheadersize;
  uint32_t width;
  uint32_t height;
  uint16_t planes;
  uint16_t bitsperpixel;
  uint32_t compression;
  uint32_t imagesize;
  uint32_t ypixelpermeter;
  uint32_t xpixelpermeter;
  uint32_t numcolorspallette;
  uint32_t mostimpcolor;
} BitmapInfoHeader;
typedef struct
{
  Fileheader fileheader;
  BitmapInfoHeader bitmapinfoheader;
} Bitmap;
#pragma pack(pop)

void genBpm(const char *filename, int height, int width, int *pixelbuffer_i)
{
  uint32_t pixelbytesize = height * width * _bitsperpixel / 8;
  uint32_t _filesize = pixelbytesize + sizeof(Bitmap);

  FILE *fp = fopen(filename, "wb");
  Bitmap *pbitmap = (Bitmap *)calloc(1, sizeof(Bitmap));

  int buffer_size = height * width * 4;
  uint8_t *pixelbuffer = (uint8_t *)malloc(buffer_size);

  for (int i = 0; i < buffer_size; i++)
  {
    pixelbuffer[i] = (uint8_t)pixelbuffer_i[i];
  }

  pbitmap->fileheader.signature[0] = 'B';
  pbitmap->fileheader.signature[1] = 'M';
  pbitmap->fileheader.filesize = _filesize;
  pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(Bitmap);
  pbitmap->bitmapinfoheader.dibheadersize = sizeof(BitmapInfoHeader);
  pbitmap->bitmapinfoheader.width = width;
  pbitmap->bitmapinfoheader.height = height;
  pbitmap->bitmapinfoheader.planes = _planes;
  pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
  pbitmap->bitmapinfoheader.compression = _compression;
  pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
  pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
  pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
  pbitmap->bitmapinfoheader.numcolorspallette = 0;

  fwrite(pbitmap, 1, sizeof(Bitmap), fp);
  fwrite(pixelbuffer, 1, pixelbytesize, fp);
  fclose(fp);

  free(pbitmap);
  free(pixelbuffer);
}