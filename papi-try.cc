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
#include <pthread.h>
#include <unistd.h>

#include <mpi.h>

/* max events to monitor */
#define MAX_EVENTS 16

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
  int size;                      /* world size (from MPI) */
  int timeout;                   /* alarm timeout */
  const char* names[MAX_EVENTS]; /* names of the events to monitor */
  int n;                         /* number of events */
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
 * PAPI helpers
 */
static void PAPI_complain(int err, const char* msg) {
  complain(EXIT_FAILURE, 0, "PAPI %s: %s", msg, PAPI_strerror(err));
}

/*
 * setup PAPI for performance monitoring.
 */
static int PAPI_prepare(int EventSet) {
  int tmp[MAX_EVENTS];
  int i, rv;

  for (i = 0; i < g.n; i++) {
    rv = PAPI_event_name_to_code(g.names[i], &tmp[i]);
    if (rv != PAPI_OK) {
      PAPI_complain(rv, g.names[i]);
    }
  }

  rv = PAPI_add_events(EventSet, tmp, g.n);
  if (rv != PAPI_OK) {
    PAPI_complain(rv, "add events");
  }

  return EventSet;
}

/*
 * start monitoring.
 */
static void PAPI_run(int EventSet) {
  int rv = PAPI_start(EventSet);
  if (rv != PAPI_OK) PAPI_complain(rv, "start");
}

/*
 * reset all counters.
 */
static void PAPI_clear(int EventSet) {
  int rv = PAPI_reset(EventSet);
  if (rv != PAPI_OK) PAPI_complain(rv, "reset");
}

/*
 * read current counter value.
 */
static void PAPI_fetch(int EventSet, long long* value) {
  int rv = PAPI_read(EventSet, value);
  if (rv != PAPI_OK) PAPI_complain(rv, "read");
}

/*
 * report
 */
static void report(const long long* value) {
  printf("\n");
  for (int i = 0; i < g.n; i++) {
    printf("%s: %lld\n", g.names[i], value[i]);
  }
  printf("\n");
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
static int runops(size_t mb);
static void doit();

/*
 * main program.
 */
int main(int argc, char* argv[]) {
  int ch;

  argv0 = argv[0];

  /* MPI wants us to call this early as possible */
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    complain(EXIT_FAILURE, 1, "MPI_Init failed.  MPI is required.");
  }

  /* We want lines even if we are writing to a pipe */
  setlinebuf(stdout);

  memset(&g, 0, sizeof(g));

  if (MPI_Comm_rank(MPI_COMM_WORLD, &myrank) != MPI_SUCCESS)
    complain(EXIT_FAILURE, 0, "unable to get MPI rank");
  if (MPI_Comm_size(MPI_COMM_WORLD, &g.size) != MPI_SUCCESS)
    complain(EXIT_FAILURE, 0, "unable to get MPI size");

  g.timeout = DEF_TIMEOUT;

  g.names[0] = "PAPI_L1_DCM";
  g.names[1] = "PAPI_L1_DCA";
  g.names[2] = "PAPI_L2_DCM";
  g.names[3] = "PAPI_L2_DCA";

  g.n = 4;

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

  if (argc != 0) {
    for (int i = 0; i < argc; i++) {
      g.names[i] = argv[i];
    }
    g.n = argc;
  }

  if (myrank == 0) {
    printf("== Events:\n");
    for (int i = 0; i < g.n; i++) printf(" > %s\n", g.names[i]);
    printf("\n");
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
  int EventSet = PAPI_NULL;
  long long value[MAX_EVENTS];
  int rv;

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    complain(EXIT_FAILURE, 0, "PAPI Init failed");

  rv = PAPI_thread_init(pthread_self);
  if (rv != PAPI_OK) PAPI_complain(rv, "thread init");
  rv = PAPI_create_eventset(&EventSet);
  if (rv != PAPI_OK) PAPI_complain(rv, "create event set");

  EventSet = PAPI_prepare(EventSet);
  PAPI_run(EventSet);

  for (;;) {
    PAPI_clear(EventSet);
    runops(1 << 20);

    PAPI_fetch(EventSet, value);
    report(value);
    break;
  }

  PAPI_shutdown();
}

static int runops(size_t sz) {
  unsigned char* mem;

  mem = static_cast<unsigned char*>(malloc(sz));
  if (!mem) {
    fprintf(stderr, "Cannot alloc memory, %d MiB: %s\n", int(sz >> 20),
            strerror(errno));
    return -1;
  } else {
    for (size_t i = 0; i < sz; i++) {
      mem[rand() % sz]++;
    }
    printf("%d MiB: OK\n", int(sz >> 20));
    return 0;
  }
}
