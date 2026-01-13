/*
 * This file is derived from pymetranet
 * https://github.com/eldesradar/pymetranet
 *
 * Copyright (c) 2026, Eldes Radar
 * Licensed under the BSD 3-Clause License.
 *
 * See the LICENSE file in the root of this project for the full license text.
 */


/*==INCLUDE FILE ===============================================================*/

#ifdef DATA_SYS
#include <stdioLib.h>
#else
#include <stdio.h>
#endif


/*==bitx.h - definitions for bit streams =======================================*/

#ifndef BITIO_H
#define BITIO_H

typedef struct bit_file {
    unsigned char *addr;
    unsigned char mask;
    int rack;
    int len;
    int count;
    int pacifier_counter;
} BIT_STRM;

#ifdef __STDC__

BIT_STRM     *OpenInputBitStream( unsigned char *addr, int len );
BIT_STRM     *OpenOutputBitStream( unsigned char *addr, int len );
int           StreamOutputBit( BIT_STRM *bit_file, int bit );
int           StreamOutputBits( BIT_STRM *bit_file,
                          unsigned long code, int count );
int           StreamInputBit( BIT_STRM *bit_file );
unsigned long StreamInputBits( BIT_STRM *bit_file, int bit_count );
int          CloseInputBitStream( BIT_STRM *bit_file );
int          CloseOutputBitStream( BIT_STRM *bit_file );

#else   /* __STDC__ */

BIT_STRM     *OpenInputBitStream();
BIT_STRM     *OpenOutputBitStream();
int           StreamOutputBit();
int           StreamOutputBits();
int           StreamInputBit();
unsigned long StreamInputBits();
int          CloseInputBitStream();
int          CloseOutputBitStream();

#endif

#endif


/*======END======END======END======END======END======END======END======END======*/
