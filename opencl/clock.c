#include <clock.h>

#include <stdlib.h>

double WallClock (void)
{
   struct timeval  t;
   //struct timezone tz;
   //gettimeofday (&t, &tz);
   gettimeofday (&t, NULL);
   return (double) t.tv_sec + 1.0e-6 * (double) t.tv_usec;
}

void wallclock (double *t) { *t = WallClock(); }
void wallclock_ (double *t) { *t = WallClock(); }
void wallclock__ (double *t) { *t = WallClock(); }

