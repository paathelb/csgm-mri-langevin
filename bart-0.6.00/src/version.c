/* Copyright 2015-2020. Martin Uecker
 * Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2015-2020 Martin Uecker <uecker@med.uni-goettingen.de>
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/version.h"
#include "misc/debug.h"


static const char usage_str[] = "[-h]";
static const char help_str[] = 
	"Print BART version. The version string is of the form\n"
	"TAG or TAG-COMMITS-SHA as produced by 'git describe'. It\n"
	"specifies the last release (TAG), and (if git is used)\n"
	"the number of commits (COMMITS) since this release and\n"
	"the abbreviated hash of the last commit (SHA). If there\n"
	"are local changes '-dirty' is added at the end.\n";


static bool parse_version(const char* version, unsigned int v[5])
{
	int q, r, s;

	int len = strlen(version);

	v[3] = 0;	// patch level

	int ret = sscanf(version, "v%u.%u.%u%n-%u-g%*7[0-9a-f]%n-dirty%n", &v[0], &v[1], &v[2], &q, &v[3], &r, &s);

	if (!(   ((3 == ret) && (len == q))
	      || ((4 == ret) && (len == r))
	      || ((4 == ret) && (len == s))))
		return false;

	for (int i = 0; i < 4; i++)
		if (v[i] >= 1000000)
			return false;

	v[4] = (len == s) ? 1 : 0;	// dirty

	return true;
}
			

int main_version(int argc, char* argv[])
{
	bool verbose = false;
	const char* version = NULL;

	const struct opt_s opts[] = {

		OPT_STRING('t', &version, "version", "Check minimum version"),
		OPT_SET('V', &verbose, "Output verbose info"),
	};

	cmdline(&argc, argv, 0, 0, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != version) {

		unsigned int va[5];
		unsigned int vb[5];

		if (!parse_version(bart_version, va))
			assert(0);

		if (!parse_version(version, vb))
			error("Version number not recognized.\n");

		if (0 < vb[4]) { // dirty

			if (0 == strcmp(bart_version, version))
				return 0;

			debug_printf(DP_WARN, "Comparing to dirty version. Exact match required.\n");

			return 1;
		}

		for (int i = 0; i < 5; i++) { // lexicographical comparison

			if (va[i] > vb[i])
				return 0;

			if (va[i] < vb[i])
				return 1;
		}

		return 0;
	}


	bart_printf("%s\n", bart_version);

	if (verbose) {

#ifdef __GNUC__
		bart_printf("GCC_VERSION=%s\n", __VERSION__);
#endif

		bart_printf("CUDA=");
#ifdef USE_CUDA
			bart_printf("1\n");
#else
			bart_printf("0\n");
#endif

		bart_printf("ACML=");
#ifdef USE_ACML
			bart_printf("1\n");
#else
			bart_printf("0\n");
#endif

		bart_printf("FFTWTHREADS=");
#ifdef FFTWTHREADS
			bart_printf("1\n");
#else
			bart_printf("0\n");
#endif
	}

	return 0;
}



