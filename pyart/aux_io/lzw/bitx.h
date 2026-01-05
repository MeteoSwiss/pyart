/*-------------------------------------------------------------------------------\
|										 |
|  bitx.h - Include files for bit-wise data manipulation	      20-May-93	 |
|										 |
|--------------------------------------------------------------------------------|
|										 |
|     This program is copyright 1993 by Lassen Research, Manton, CA 96059,	 |
|     USA, all rights reserved.  It is intended for use only on a specific	 |
|     customer processor and is not to be transferred or otherwise divulged	 |
|     to third parties without the written permission of Lassen Research.	 |
|     This program may be copied or modified by the customer for use on the	 |
|     licensed processor, provided that this copyright notice is included.	 |
|  										 |
|--------------------------------------------------------------------------------|
|										 |
|  Usage:									 |
|										 |
|	#include bitx.h"	Use as include file for lzw.c			 |
|										 |
|  Processing:									 |
|										 |
|	Define bit_file structure for handling bit stream and function names	 |
|										 |
|  Version history:								 |
|										 |
|	V1.0	20-Nov-92	KenB 	Origin					 |
|	V1.3	20-May-93	CWJiang Add header  				 |
|										 |
|--------------------------------------------------------------------------------|
|										 |
|  Header information:								 |
|										 |
|	Software suite:		Swiss Radar Site				 |
|	Package:		Head file       				 |
|	Source file:		/project/SRN/incude/bitx.h			 |
|	Release state:		$State: Exp $					 |
|	Revision number:	$Revision: 1.1.1.1 $				 |
|	Revised by:		$Author: jiang $				 |
|	Revision date:		$Date: 2009/09/12 00:07:28 $			 |
|										 |
\-------------------------------------------------------------------------------*/


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
