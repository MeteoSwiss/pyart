/*
 * This file is derived from pymetranet
 * https://github.com/eldesradar/pymetranet
 *
 * Copyright (c) 2026, Eldes Radar
 * Licensed under the BSD 3-Clause License.
 *
 * See the LICENSE file in the root of this project for the full license text.
 */
<<<<<<< HEAD
=======

>>>>>>> f70b7b3aec807bf662026b8511428deb8018010e

/*==INCLUDE FILE ===============================================================*/

#ifndef DATA_SYS

#include <stdio.h>
#include <stdlib.h>

#endif
#include "bitx.h"

/*==DEFINES ====================================================================*/

#define PACIFIER_COUNT 2047
#define END_OF_STREAM	256

/*==EXTERNAL DEFINITIONS =======================================================*/



/*==bitx() - bit-wise input and output routines ================================*/



/*----------------------------------------------------------------------*/
/* Global Variables:							*/
/*----------------------------------------------------------------------*/
extern int lzw_verbose;

/*----------------------------------------------------------------------*/
void dump_bit_file(bf)
BIT_STRM *bf;
{
    fprintf(stderr, "\naddr      :%p\n", (void*)bf->addr);
    fprintf(stderr, "mask        :%02x\n", bf->mask);
    fprintf(stderr, "rack        :%08x\n", bf->rack);
    fprintf(stderr, "len         :%d\n", bf->len);
    fprintf(stderr, "count       :%d\n", bf->count);
    fprintf(stderr, "pacifier_cou:%d\n", bf->pacifier_counter);
}


/*----------------------------------------------------------------------*/
BIT_STRM *OpenOutputBitStream( name, len )
unsigned char *name;
int len;
{
	BIT_STRM *bit_strm;

	bit_strm = (BIT_STRM *) calloc( 1, sizeof( BIT_STRM ) );
	if ( bit_strm == NULL )
		return( NULL);
	bit_strm->addr = name;
	bit_strm->rack = 0;
	bit_strm->mask = 0x80;
	bit_strm->len = len;
	bit_strm->count = 0;
	bit_strm->pacifier_counter = 0;
	if(lzw_verbose > 2)
	{
		fprintf(stderr, "\nOpenOutputBitStream:");
		dump_bit_file(bit_strm);
	}
	return( bit_strm );
}

/*----------------------------------------------------------------------*/
BIT_STRM *OpenInputBitStream( name, len )
unsigned char *name;
int len;
{
	BIT_STRM *bit_strm;

	bit_strm = (BIT_STRM *) calloc( 1, sizeof( BIT_STRM ) );
	if ( bit_strm == NULL )
		return( NULL);
	bit_strm->addr = name;
	bit_strm->rack = 0;
	bit_strm->count = 0;
	bit_strm->mask = 0x80;
	bit_strm->len = len;
	bit_strm->pacifier_counter = 0;
	if(lzw_verbose > 2)
	{
		fprintf(stderr, "\nOpenInputBitStream:");
		dump_bit_file(bit_strm);
	}
	return( bit_strm );
}

/*----------------------------------------------------------------------*/
int nextc(bit_strm)
BIT_STRM *bit_strm;
{
	if(bit_strm->len == bit_strm->count)
		return EOF;

	bit_strm->count++;

	return (int)*bit_strm->addr++;
}

/*----------------------------------------------------------------------*/
int outc(c, bit_strm)
BIT_STRM *bit_strm;
unsigned char c;
{
	if(bit_strm->len == bit_strm->count)
		return EOF;

	bit_strm->count++;

	return (int)(*bit_strm->addr++ = c);
}


/*----------------------------------------------------------------------*/
int CloseOutputBitStream( bit_strm )
BIT_STRM *bit_strm;
{
        int i;
	if ( bit_strm->mask != 0x80 )
		if ( outc( bit_strm->rack, bit_strm ) != bit_strm->rack )
		{
			/*fprintf(stderr, "Fatal error in CloseBitStream!\n" );*/
			return -1;
		}
	if(lzw_verbose > 2)
	{
		fprintf(stderr, "\nCloseOutputBitStream:");
		dump_bit_file(bit_strm);
	}
	i=bit_strm->count;
	free( (char *) bit_strm );
	/*
	return bit_strm->count;
	*/
	return i;
}

/*----------------------------------------------------------------------*/
int CloseInputBitStream( bit_strm )
BIT_STRM *bit_strm;
{
        int i;
	if(lzw_verbose > 2)
	{
		fprintf(stderr, "\nCloseInputBitStream:");
		dump_bit_file(bit_strm);
	}
	i=bit_strm->count;
	free( (char *) bit_strm );
	/*
	return bit_strm->count;
	*/
	return i;
}

/*----------------------------------------------------------------------*/
int StreamOutputBit( bit_strm, bit )
BIT_STRM *bit_strm;
int bit;
{
	if ( bit )
		bit_strm->rack |= bit_strm->mask;
	bit_strm->mask >>= 1;
	if ( bit_strm->mask == 0 )
	{
		if ( outc( bit_strm->rack, bit_strm ) != bit_strm->rack )
		{
			/*fprintf(stderr, "Fatal error in StreamOutputBit!\n" );*/
			return -1;
		}
		else
			if(((bit_strm->pacifier_counter++ & PACIFIER_COUNT)
				== 0) && lzw_verbose)
				putc( '.', stderr );
		bit_strm->rack = 0;
		bit_strm->mask = 0x80;
	}
	return 0;
}

/*----------------------------------------------------------------------*/
int StreamOutputBits( bit_strm, code, count )
BIT_STRM *bit_strm;
unsigned long code;
int count;
{
	unsigned long mask;

	mask = 1L << ( count - 1 );
	while ( mask != 0) {
		if ( mask & code )
			bit_strm->rack |= bit_strm->mask;
		bit_strm->mask >>= 1;
		if ( bit_strm->mask == 0 )
		{
			if( outc( bit_strm->rack, bit_strm ) != bit_strm->rack )
			{
				/*fprintf(stderr,
				    "Fatal error in StreamOutputBits!\n" );*/
				return -1;
			}
			else if(((bit_strm->pacifier_counter++&PACIFIER_COUNT)
					==0) && lzw_verbose)
				putc( '.', stderr );
			bit_strm->rack = 0;
			bit_strm->mask = 0x80;
		}
		mask >>= 1;
	}
	return 0;
}

/*----------------------------------------------------------------------*/
int StreamInputBit( bit_strm )
BIT_STRM *bit_strm;
{
	int value;

	if ( bit_strm->mask == 0x80 ) {
		bit_strm->rack = nextc( bit_strm );
		if ( bit_strm->rack == EOF )
		{
			fprintf(stderr, "Fatal error in StreamInputBit!\n" );
			return END_OF_STREAM;
		}
		if (( ( bit_strm->pacifier_counter++ & PACIFIER_COUNT ) == 0 )
			&& lzw_verbose)
			putc( '.', stderr );
	}
	value = bit_strm->rack & bit_strm->mask;
	bit_strm->mask >>= 1;
	if ( bit_strm->mask == 0 )
		bit_strm->mask = 0x80;
	return( value ? 1 : 0 );
}

/*----------------------------------------------------------------------*/
unsigned long StreamInputBits( bit_strm, bit_count )
BIT_STRM *bit_strm;
int bit_count;
{
	unsigned long mask;
	unsigned long return_value;

	mask = 1L << ( bit_count - 1 );
	return_value = 0;
	while ( mask != 0) {
		if ( bit_strm->mask == 0x80 )
		{
			bit_strm->rack = nextc( bit_strm );
			if ( bit_strm->rack == EOF )
			{
				fprintf(stderr,
					"Fatal error in StreamInputBit!\n" );
				return END_OF_STREAM;
			}
			if(( ( bit_strm->pacifier_counter++ & PACIFIER_COUNT )
				== 0 ) && lzw_verbose)
				putc( '.', stderr );
		}
		if ( bit_strm->rack & bit_strm->mask )
			return_value |= mask;
		mask >>= 1;
		bit_strm->mask >>= 1;
		if ( bit_strm->mask == 0 )
			bit_strm->mask = 0x80;
	}
	return( return_value );
}
/*************************** End of BITIO.C **************************/
/*-END OF MODULE--------------------------------------------------------*/
