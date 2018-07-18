/*
 * Copyright (c) 2018, Carnegie Mellon University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * papi-try.cc  run papi and numa to check memory configurations
 * on other people's machines
 */

#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <numa.h>
#include <papi.h>
#include <unistd.h>

#include <mpi.h>

/*
 * helper/utility functions, included inline here so we are self-contained
 * in one single source file...
 */
static char* argv0; /* argv[0], program name */
static int myrank = 0;

/*
 * vcomplain/complain about something.  if ret is non-zero we exit(ret)
 * after complaining.  if r0only is set, we only print if myrank == 0.
 */
static void vcomplain(int ret, int r0only, const char* format, va_list ap) {
  if (!r0only || myrank == 0) {
    fprintf(stderr, "%s: ", argv0);
    vfprintf(stderr, format, ap);
    fprintf(stderr, "\n");
  }
  if (ret) {
    MPI_Finalize();
    exit(ret);
  }
}

static void complain(int ret, int r0only, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  vcomplain(ret, r0only, format, ap);
  va_end(ap);
}

/*
 * default values
 */
#define DEF_TIMEOUT 120 /* alarm timeout */

/*
 * gs: shared global data (e.g. from the command line)
 */
static struct gs {
  int size;    /* world size (from MPI) */
  int timeout; /* alarm timeout */
} g;

/*
 * alarm signal handler
 */
static void sigalarm(int foo) {
  fprintf(stderr, "SIGALRM detected (%d)\n", myrank);
  fprintf(stderr, "Alarm clock\n");
  MPI_Finalize();
  exit(1);
}

/*
 * usage
 */
static void usage(const char* msg) {
  /* only have rank 0 print usage error message */
  if (myrank) goto skip_prints;

  if (msg) fprintf(stderr, "%s: %s\n", argv0, msg);
  fprintf(stderr, "usage: %s [options]\n", argv0);
  fprintf(stderr, "\noptions:\n");
  fprintf(stderr, "\t-t sec      timeout (alarm), in seconds\n");

skip_prints:
  MPI_Finalize();
  exit(1);
}

/*
 * forward prototype decls.
 */
static void doit();

/*
 * main program.
 */

int main(int argc, char* argv[]) {
  int ch;

  argv0 = argv[0];

  /* MPI wants us to call this early as possible */
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    complain(EXIT_FAILURE, 1, "%s: MPI_Init failed.  MPI is required.", argv0);
  }

  /* We want lines even if we are writing to a pipe */
  setlinebuf(stdout);

  memset(&g, 0, sizeof(g));

  if (MPI_Comm_rank(MPI_COMM_WORLD, &myrank) != MPI_SUCCESS)
    complain(EXIT_FAILURE, 0, "unable to get MPI rank");
  if (MPI_Comm_size(MPI_COMM_WORLD, &g.size) != MPI_SUCCESS)
    complain(EXIT_FAILURE, 0, "unable to get MPI size");

  g.timeout = DEF_TIMEOUT;

  while ((ch = getopt(argc, argv, "t:")) != -1) {
    switch (ch) {
      case 't':
        g.timeout = atoi(optarg);
        if (g.timeout < 0) usage("bad timeout");
        break;
      default:
        usage(NULL);
    }
  }

  argc -= optind;
  argv += optind;

  if (myrank == 0) {
    printf("== Program options:\n");
    printf(" > MPI_rank   = %d\n", myrank);
    printf(" > MPI_size   = %d\n", g.size);
    printf(" > timeout    = %d secs\n", g.timeout);
    printf("\n");
  }

  signal(SIGALRM, sigalarm);
  alarm(g.timeout);

  doit();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}

static void doit() {
  //
}
