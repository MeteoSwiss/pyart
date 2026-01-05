/*lzw.h
*/

int Compress(unsigned char *inputaddr, int insize, unsigned char *outputaddr, int outsize);
int Expand(unsigned char *inputaddr, int insize, unsigned char *outputaddr, int outsize);
