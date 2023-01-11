#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 600000000ULL

/* その他付属関数のプロトタイプ宣言 */
void snprintf_time(char *buf, size_t size, struct tm tm, struct timespec ts);
void print_elapsed(struct timespec ts, struct timespec te);

void initialize_array(float *arr)
{
    for (unsigned long long i = 0; i < N; i++)
    {
        *(arr + i) = (float)i;
    }
    return;
}

void sum_of_array(float *arr1, float *arr2, float *arr3, float *sum_arr)
{
    for (unsigned long long i = 0; i < N; i++)
    {
        *(sum_arr + i) = *(arr1 + i) + *(arr2 + i) + *(arr3 + i);
    }
    return;
}

void check_value(float *arr1, float *arr2, float *arr3, float *sum_arr, unsigned long long index)
{
    printf("INDEX=%ld: %f = %f + %f + %f\n", index, *(sum_arr + index), *(arr1 + index), *(arr2 + index), *(arr3 + index));
    return;
}

int main(void)
{
    float *arr1, *arr2, *arr3, *sum_arr;
    size_t n_byte = N * sizeof(float);
    struct timespec ts, te;

    /* initialize */
    printf("malloc for arrays.\n");
    arr1 = (float *)malloc(n_byte);
    arr2 = (float *)malloc(n_byte);
    arr3 = (float *)malloc(n_byte);
    sum_arr = (float *)malloc(n_byte);

    printf("initialize arrays.\n");
    initialize_array(arr1);
    initialize_array(arr2);
    initialize_array(arr3);

    /* main */
    printf("start calculation process\n");
    clock_gettime(CLOCK_REALTIME, &ts);
    sum_of_array(arr1, arr2, arr3, sum_arr);
    clock_gettime(CLOCK_REALTIME, &te);
    printf("finish calculation process\n");

    /* result */
    print_elapsed(ts, te);

    check_value(arr1, arr2, arr3, sum_arr, 0);
    check_value(arr1, arr2, arr3, sum_arr, 100);
    check_value(arr1, arr2, arr3, sum_arr, N - 1);
}

/* ----------------------------- */
/* その他の付属関数本体              */
/* ----------------------------- */
void snprintf_time(char *buf, size_t size, struct tm tm, struct timespec ts)
{

    int n = snprintf(buf,
                     size,
                     "%d/%02d/%02d %02d:%02d:%02d.%09ld",
                     tm.tm_year + 1900,
                     tm.tm_mon + 1,
                     tm.tm_mday,
                     tm.tm_hour,
                     tm.tm_min,
                     tm.tm_sec,
                     ts.tv_nsec);
    return;
}

void print_elapsed(struct timespec ts, struct timespec te)
{
    long long elapsed;
    struct tm tms, tme;
    char start_time[64], end_time[64];

    /* convert to local time */
    localtime_r(&ts.tv_sec, &tms);
    localtime_r(&te.tv_sec, &tme);

    /* calculate */
    elapsed = (te.tv_sec - ts.tv_sec) * 1000000000LL + (te.tv_nsec - ts.tv_nsec);
    long long e_sec = elapsed / 1000000000LL;
    long long e_usec = elapsed - e_sec * 1000000000LL;

    /* print result */
    snprintf_time(start_time, sizeof(start_time), tms, ts);
    snprintf_time(end_time, sizeof(end_time), tme, te);

    printf("\n\n<<<<<<< result >>>>>>>>>>>\n");
    printf("Start time   = %s\n", start_time);
    printf("End time     = %s\n", end_time);
    printf("Elapsed time = %ld.%09ld sec\n", e_sec, e_usec);

    return;
}