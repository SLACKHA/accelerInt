#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <clock.h>
#include <cklib.h>
#include <rk.h>
#include <ros.h>

#define CL_EXEC(__cmd__) {\
      cl_uint _ret = (__cmd__); \
      if (_ret != CL_SUCCESS) \
      { \
         fprintf(stderr,"Error executing CL cmd =\n\t%s\n", #__cmd__); \
         fprintf(stderr,"\tret  = %d\n", _ret); \
         fprintf(stderr,"\tline = %d\n", __LINE__); \
         fprintf(stderr,"\tfile = %s\n", __FILE__); \
         exit(-1); \
      } \
   }

#define __clerror(__errcode) \
   { \
      fprintf(stderr,"errcode = %d ", (__errcode)); \
      if ((__errcode) == CL_INVALID_CONTEXT) \
         fprintf(stderr,"CL_INVALID_CONTEXT\n"); \
      else if ((__errcode) == CL_INVALID_VALUE) \
         fprintf(stderr,"CL_INVALID_VALUE\n"); \
      else if ((__errcode) == CL_INVALID_BUFFER_SIZE) \
         fprintf(stderr,"CL_INVALID_BUFFER_SIZE\n"); \
      else if ((__errcode) == CL_INVALID_HOST_PTR) \
         fprintf(stderr,"CL_INVALID_HOST_PTR\n"); \
      else if ((__errcode) == CL_MEM_OBJECT_ALLOCATION_FAILURE) \
         fprintf(stderr,"CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); \
      else if ((__errcode) == CL_OUT_OF_RESOURCES) \
         fprintf(stderr,"CL_OUT_OF_RESOURCES\n"); \
      else if ((__errcode) == CL_OUT_OF_HOST_MEMORY) \
         fprintf(stderr,"CL_OUT_OF_HOST_MEMORY\n"); \
   }

int isPower2 (int x)
{
   return ( (x > 0) && ((x & (x-1)) == 0) );
}
size_t load_source_from_file (const char *flname, char *source_str, size_t source_size)
{
   FILE *fp = fopen(flname, "rb");
   assert (fp != NULL);

   fseek (fp, 0, SEEK_END);
   size_t sz = ftell(fp);
   rewind (fp);

   assert (sz < source_size);

   sz = fread(source_str, sizeof(char), source_size, fp);

   fclose(fp);

   return sz;
}
int imax(int a, int b) { return (a > b) ? a : b; }
int imin(int a, int b) { return (a < b) ? a : b; }

size_t write_ckdata_table (const ckdata_t *ck, char *ckobj_name, char *source_str, size_t max_source_size)
{
   size_t sz = 0;

   const int kk = ck->n_species;
   const int ii = ck->n_reactions;

   //size_t max_source_size = 0x80000; // 8 MB
   //char *str = malloc(sizeof(char)*max_source_size);
   char *str = source_str;

   //sz += sprintf(str + sz, "\n#include <cklib.h>\n");
   sz += sprintf(str + sz, "\n#define __ckobj_name__ %s\n", ckobj_name);

   if (0)
   {
      sz += sprintf(str + sz, "\n__constant ckdata_t %s = \n", ckobj_name);
   }
   else
   {
      sz += sprintf(str + sz, "\n__constant struct %s_s\n", ckobj_name);
      sz += sprintf(str + sz, "{\n");
      sz += sprintf(str + sz, "   int n_species;\n");
      sz += sprintf(str + sz, "   double sp_mwt[%d];\n", kk);
      sz += sprintf(str + sz, "   double th_tmid[%d];\n", kk);
      sz += sprintf(str + sz, "   double th_alo[%d][__ck_max_th_terms];\n", kk);
      sz += sprintf(str + sz, "   double th_ahi[%d][__ck_max_th_terms];\n", kk);
      sz += sprintf(str + sz, "   int n_reactions;\n");
      sz += sprintf(str + sz, "   double rx_A[%d];\n", ii);
      sz += sprintf(str + sz, "   double rx_b[%d];\n", ii);
      sz += sprintf(str + sz, "   double rx_E[%d];\n", ii);
      sz += sprintf(str + sz, "   int rx_nu[%d][__ck_max_rx_order*2];\n", ii);
      sz += sprintf(str + sz, "   int rx_nuk[%d][__ck_max_rx_order*2];\n", ii);
      sz += sprintf(str + sz, "   int rx_sumnu[%d];\n", ii);
      sz += sprintf(str + sz, "   int n_reversible_reactions;\n");
      sz += sprintf(str + sz, "   int rx_rev_idx[%d];\n", imax(1,ck->n_reversible_reactions));
      sz += sprintf(str + sz, "   double rx_rev_A[%d];\n", imax(1,ck->n_reversible_reactions));
      sz += sprintf(str + sz, "   double rx_rev_b[%d];\n", imax(1,ck->n_reversible_reactions));
      sz += sprintf(str + sz, "   double rx_rev_E[%d];\n", imax(1,ck->n_reversible_reactions));
      sz += sprintf(str + sz, "   int n_irreversible_reactions;\n");
      sz += sprintf(str + sz, "   int rx_irrev_idx[%d];\n", ck->n_irreversible_reactions);
      sz += sprintf(str + sz, "   int n_thdbdy;\n");
      sz += sprintf(str + sz, "   int rx_thdbdy_idx[%d];\n", ck->n_thdbdy);
      sz += sprintf(str + sz, "   int rx_thdbdy_offset[%d];\n", ck->n_thdbdy+1);
      sz += sprintf(str + sz, "   int rx_thdbdy_spidx[%d];\n", ck->rx_thdbdy_offset[ck->n_thdbdy]);
      sz += sprintf(str + sz, "   double rx_thdbdy_alpha[%d];\n", ck->rx_thdbdy_offset[ck->n_thdbdy]);
      sz += sprintf(str + sz, "   int n_falloff;\n");
      sz += sprintf(str + sz, "   int rx_falloff_idx[%d];\n", ck->n_falloff);
      sz += sprintf(str + sz, "   int rx_falloff_spidx[%d];\n", ck->n_falloff);
      sz += sprintf(str + sz, "   double rx_falloff_params[%d][__ck_max_falloff_params];\n", ck->n_falloff);
      sz += sprintf(str + sz, "   int rx_info[%d];\n", ii);
      sz += sprintf(str + sz, "} %s = \n", ckobj_name);
   }
   sz += sprintf(str + sz, "{\n");

   #define __write_array1(__a, __format, __n) \
      sz += sprintf(str + sz, "   .%s = {", #__a); \
      for (int i = 0; i < (__n); ++i) \
      { \
         sz += sprintf(str + sz, __format, ck-> __a [i]); \
         if (i != (__n)-1) \
            sz += sprintf(str + sz, ", "); \
      } \
      sz += sprintf(str + sz, "},\n");

   #define __write_array2(__a, __format, __m, __n) \
   sz += sprintf(str + sz, "   .%s = {", #__a); \
      for (int k = 0; k < (__m); ++k) \
      { \
         sz += sprintf(str + sz, "\t\t{"); \
         for (int i = 0; i < (__n); ++i) \
         { \
            sz += sprintf(str + sz, __format, ck-> __a [k][i]); \
            if (i != (__n)-1) \
               sz += sprintf(str + sz, ", "); \
         } \
         sz += sprintf(str + sz, "}"); \
         if (k != (__m)-1) \
            sz += sprintf(str + sz, ",\n"); \
      } \
      sz += sprintf(str + sz, "},\n");

   sz += sprintf(str + sz, "   .n_species = %d,\n", kk);

   char *dp_format = "%a";

   __write_array1( sp_mwt, dp_format, kk);

   __write_array1( th_tmid, dp_format, kk);
   __write_array2( th_alo, dp_format, kk, __ck_max_th_terms);
   __write_array2( th_ahi, dp_format, kk, __ck_max_th_terms);

   sz += sprintf(str + sz, "   .n_reactions = %d,\n", ii);

   __write_array1(rx_A, dp_format, ii)
   __write_array1(rx_b, dp_format, ii)
   __write_array1(rx_E, dp_format, ii)

   __write_array2( rx_nu, "%d", ii, __ck_max_rx_order*2);
   __write_array2( rx_nuk, "%d", ii, __ck_max_rx_order*2);
   __write_array1( rx_sumnu, "%d", ii)

   sz += sprintf(str + sz, "   .n_reversible_reactions = %d,\n", ck->n_reversible_reactions);

   if (ck->n_reversible_reactions > 0)
   {
      __write_array1(rx_rev_idx, "%d", ck->n_reversible_reactions)
      __write_array1(rx_rev_A, dp_format, ck->n_reversible_reactions)
      __write_array1(rx_rev_b, dp_format, ck->n_reversible_reactions)
      __write_array1(rx_rev_E, dp_format, ck->n_reversible_reactions)
   }

   sz += sprintf(str + sz, "   .n_irreversible_reactions = %d,\n", ck->n_irreversible_reactions);

   if (ck->n_irreversible_reactions > 0)
   {
      __write_array1(rx_irrev_idx, "%d", ck->n_irreversible_reactions)
   }

   sz += sprintf(str + sz, "   .n_thdbdy = %d,\n", ck->n_thdbdy);

   if (ck->n_thdbdy > 0)
   {
      __write_array1( rx_thdbdy_idx, "%d", ck->n_thdbdy);
      __write_array1( rx_thdbdy_offset, "%d", ck->n_thdbdy+1);
      __write_array1( rx_thdbdy_spidx, "%d", ck->rx_thdbdy_offset[ck->n_thdbdy]);
      __write_array1( rx_thdbdy_alpha, dp_format, ck->rx_thdbdy_offset[ck->n_thdbdy]);
   }

   sz += sprintf(str + sz, "   .n_falloff = %d,\n", ck->n_falloff);

   if (ck->n_falloff > 0)
   {
      __write_array1( rx_falloff_idx, "%d", ck->n_falloff);
      __write_array1( rx_falloff_spidx, "%d", ck->n_falloff);
      __write_array2( rx_falloff_params, dp_format, ck->n_falloff, __ck_max_falloff_params);
   }

   __write_array1(rx_info, "%x", ii)
      //sz += sprintf(str + sz, "}\n"); // no comma!

   sz += sprintf(str + sz, "};\n");

   if (0)
   {
      char *flname = malloc(sizeof(char) * strlen(ckobj_name) + 4);
      strcpy (flname, ckobj_name);
      strcat (flname, ".cl");

      FILE *fp = fopen(flname,"w");
      if (fp == NULL)
      {
         fprintf(stderr,"Error opening %s\n", flname);
         exit(-1);
      }

      fprintf(fp,"%s\n", str);

      fclose(fp);
      free(flname);
   }

   //free(str);

   return sz;
}

#define __Alignment (128)

cl_mem CreateBuffer (cl_context *context, cl_mem_flags flags, size_t size, void *host_ptr)
{
   cl_int ret;
   cl_mem buf = clCreateBuffer (*context, flags, size + __Alignment, host_ptr, &ret);
   //assert( ret == CL_SUCCESS );
   if (ret != CL_SUCCESS)
   {
      fprintf(stderr," Error in CreateBuffer: flags=%d, size=%lu, host_ptr=%x\n", flags, size, host_ptr);
      __clerror(ret);
      exit(-1);
   }

   return buf;
}

cl_device_type getDeviceType (cl_device_id device_id)
{
   cl_device_type val;
   CL_EXEC( clGetDeviceInfo (device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &val, NULL) );

   return val;
}

typedef struct //kernelInfo_s
{
   char function_name[1024];
   char attributes[1024];
   cl_uint num_args;
   cl_uint reference_count;
   size_t compile_work_group_size[3];
   size_t work_group_size;
   size_t preferred_work_group_size_multiple;
   cl_ulong local_mem_size;
   size_t global_work_size[3];
   cl_ulong private_mem_size;
}
kernelInfo_t;

void printKernelInfo (kernelInfo_t *info)
{
   printf("Kernel Info:\n");
   printf("\tfunction_name = %s\n", info->function_name);
   #if (__OPENCL_VERSION__ >= 120)
   printf("\tattributes = %s\n", info->attributes);
   #endif
   printf("\tnum_args = %d\n", info->num_args);
   printf("\treference_count = %d\n", info->reference_count);
   printf("\tcompile_work_group_size = (%d,%d,%d)\n", info->compile_work_group_size[0], info->compile_work_group_size[1], info->compile_work_group_size[2]);
   printf("\twork_group_size = %d\n", info->work_group_size);
   printf("\tpreferred_work_group_size_multiple = %d\n", info->preferred_work_group_size_multiple);
   //printf("\tglobal_work_size = (%d,%d,%d)\n", info->global_work_size[0], info->global_work_size[1], info->global_work_size[2]);
   printf("\tlocal_mem_size = %d\n", info->local_mem_size);
   printf("\tprivate_mem_size = %d\n", info->private_mem_size);
}
void getKernelInfo (kernelInfo_t *info, cl_kernel kernel, cl_device_id device_id)
{
   CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_FUNCTION_NAME, sizeof(info->function_name), info->function_name, NULL) );

   #if (__OPENCL_VERSION__ >= 120)
   CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_ATTRIBUTES, sizeof(info->attributes), info->attributes, NULL) );
   #endif

   CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_NUM_ARGS, sizeof(info->num_args), &info->num_args, NULL) );

   CL_EXEC( clGetKernelInfo (kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(info->reference_count), &info->reference_count, NULL) );

   CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(info->compile_work_group_size), info->compile_work_group_size, NULL) );

   CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(info->work_group_size), &info->work_group_size, NULL) );

   CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(info->preferred_work_group_size_multiple), &info->preferred_work_group_size_multiple, NULL) );

   //CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(info->global_work_size), info->global_work_size, NULL) );

   CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(info->local_mem_size), &info->local_mem_size, NULL) );

   CL_EXEC( clGetKernelWorkGroupInfo (kernel, device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(info->private_mem_size), &info->private_mem_size, NULL) );

   printKernelInfo (info);
}

enum { cl_max_platforms = 2 };
typedef struct
{
   cl_uint num_platforms;
   cl_platform_id platform_ids[cl_max_platforms]; // All the platforms ...
   cl_platform_id platform_id; // The one we'll use ...

   char name[1024];
   char version[1024];
   char vendor[1024];
   char extensions[2048];

   int is_nvidia;
}
platformInfo_t;

void getPlatformInfo (platformInfo_t *info)
{
   cl_int ret;

   CL_EXEC ( clGetPlatformIDs ((cl_uint)cl_max_platforms, info->platform_ids, &info->num_platforms) );
   if (info->num_platforms == 0)
   {
      fprintf(stderr,"clError: num_platforms = 0\n");
      exit(-1);
   }

   info->platform_id = info->platform_ids[0];

   #define __getInfo( __STR, __VAR) { \
      CL_EXEC( clGetPlatformInfo (info->platform_id, (__STR), sizeof(__VAR), __VAR, NULL) ); \
      printf("\t%-30s = %s\n", #__STR, __VAR); }

   printf("Platform Info:\n");
   __getInfo( CL_PLATFORM_NAME,       info->name);
   __getInfo( CL_PLATFORM_VERSION,    info->version);
   __getInfo( CL_PLATFORM_VENDOR,     info->vendor);
   __getInfo( CL_PLATFORM_EXTENSIONS, info->extensions);

   #undef __getInfo

   info->is_nvidia = 0; // Is an NVIDIA device?
   if (strstr(info->vendor, "NVIDIA") != NULL)
      info->is_nvidia = 1;
   printf("\tIs-NVIDIA = %d\n", info->is_nvidia);
}

enum { cl_max_devices = 10 };
typedef struct
{
   cl_device_id device_ids[cl_max_devices];
   cl_device_id device_id;
   cl_uint num_devices;

   cl_device_type type;

   char name[1024];
   char profile[1024];
   char version[1024];
   char vendor[1024];
   char driver_version[1024];
   char opencl_c_version[1024];
   char extensions[1024];

   cl_uint native_vector_width_char;
   cl_uint native_vector_width_short;
   cl_uint native_vector_width_int;
   cl_uint native_vector_width_long;
   cl_uint native_vector_width_float;
   cl_uint native_vector_width_double;
   cl_uint native_vector_width_half;

   cl_uint preferred_vector_width_char;
   cl_uint preferred_vector_width_short;
   cl_uint preferred_vector_width_int;
   cl_uint preferred_vector_width_long;
   cl_uint preferred_vector_width_float;
   cl_uint preferred_vector_width_double;
   cl_uint preferred_vector_width_half;

   cl_uint max_compute_units;
   cl_uint max_clock_frequency;

   cl_ulong max_constant_buffer_size;
   cl_uint  max_constant_args;

   cl_ulong max_work_group_size;

   cl_ulong max_mem_alloc_size;
   cl_ulong global_mem_size;
   cl_uint  global_mem_cacheline_size;
   cl_ulong global_mem_cache_size;

   cl_device_mem_cache_type global_mem_cache_type;

   cl_ulong local_mem_size;
   cl_device_local_mem_type local_mem_type;

   cl_device_fp_config fp_config;
}
deviceInfo_t;

void getDeviceInfo (deviceInfo_t *device_info, const platformInfo_t *platform_info)
{
   const int verbose = 1;
   cl_uint ret;

   CL_EXEC( clGetDeviceIDs (platform_info->platform_id, CL_DEVICE_TYPE_ALL, (cl_uint)cl_max_devices, device_info->device_ids, &device_info->num_devices) );
   if (device_info->num_devices == 0)
   {
      fprintf(stderr,"clError: num_devices = 0\n");
      exit(-1);
   }

   char want_device_type_name[] = "CPU";
   {
      char *env = getenv("DEVICE");
      if (env)
         if (isalpha(*env))
         {
            strncpy (want_device_type_name, env, sizeof(want_device_type_name));
         }
   }
   printf("want_device_type_name = %s\n", want_device_type_name);

   device_info->device_id = device_info->device_ids[0];

   #define get_char_info(__str__, __val__, __verbose__) { \
      CL_EXEC( clGetDeviceInfo (device_info->device_id, (__str__), sizeof(__val__), __val__, NULL) ); \
      if (__verbose__) printf("\t%-40s = %s\n", #__str__, __val__); \
   }
   #define get_info(__str__, __val__, __verbose__) { \
      CL_EXEC( clGetDeviceInfo (device_info->device_id, __str__, sizeof(__val__), &__val__, NULL) ); \
      if (__verbose__) printf("\t%-40s = %ld\n", #__str__, __val__); \
   }

   printf("Device Info:\n");

   get_info( CL_DEVICE_TYPE, device_info->type, verbose);

   //if (num_devices > 1)
   {
      for (int i = 0; i < device_info->num_devices; ++i)
      {
         cl_device_type val;
         //get_info( CL_DEVICE_TYPE, val, 1);
         CL_EXEC( clGetDeviceInfo (device_info->device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &val, NULL) );
         if (verbose)
            printf("\t%-40s = %ld\n", "CL_DEVICE_TYPE", val);

         if (strstr(want_device_type_name, "GPU") || strstr(want_device_type_name, "ACC"))
         {
            if (val == CL_DEVICE_TYPE_GPU || val == CL_DEVICE_TYPE_ACCELERATOR)
            {
               device_info->device_id = device_info->device_ids[i];
               device_info->type = val;
               break;
            }
         }
      }
   }

   char device_type_name[12];
   {
      if (device_info->type == CL_DEVICE_TYPE_GPU)
         strcpy( device_type_name, "GPU");
      else if (device_info->type == CL_DEVICE_TYPE_CPU)
         strcpy( device_type_name, "CPU");
      else if (device_info->type == CL_DEVICE_TYPE_ACCELERATOR)
         strcpy( device_type_name, "ACCELERATOR");
      else if (device_info->type == CL_DEVICE_TYPE_DEFAULT)
         strcpy( device_type_name, "DEFAULT");
   }
   if (verbose) printf("\tType Name = %s\n", device_type_name);

   get_char_info( CL_DEVICE_NAME, device_info->name, verbose );
   get_char_info( CL_DEVICE_PROFILE, device_info->profile, verbose );
   get_char_info( CL_DEVICE_VERSION, device_info->version, verbose );
   get_char_info( CL_DEVICE_VENDOR, device_info->vendor, verbose );
   get_char_info( CL_DRIVER_VERSION, device_info->driver_version, verbose );
   get_char_info( CL_DEVICE_OPENCL_C_VERSION, device_info->opencl_c_version, verbose );
   //get_char_info( CL_DEVICE_BUILT_IN_KERNELS );
   get_char_info( CL_DEVICE_EXTENSIONS, device_info->extensions, verbose );

   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, device_info->native_vector_width_char, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, device_info->native_vector_width_short, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, device_info->native_vector_width_int, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, device_info->native_vector_width_long, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, device_info->native_vector_width_float, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, device_info->native_vector_width_double, verbose );
   get_info( CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, device_info->native_vector_width_half, verbose );

   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, device_info->preferred_vector_width_char, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, device_info->preferred_vector_width_short, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, device_info->preferred_vector_width_int, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, device_info->preferred_vector_width_long, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, device_info->preferred_vector_width_float, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, device_info->preferred_vector_width_double, verbose );
   get_info( CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, device_info->preferred_vector_width_half, verbose );

   get_info( CL_DEVICE_MAX_COMPUTE_UNITS, device_info->max_compute_units, verbose );
   get_info( CL_DEVICE_MAX_CLOCK_FREQUENCY, device_info->max_clock_frequency, verbose );

   get_info( CL_DEVICE_MAX_WORK_GROUP_SIZE, device_info->max_work_group_size, verbose );

   get_info( CL_DEVICE_GLOBAL_MEM_SIZE, device_info->global_mem_size, verbose );
   get_info( CL_DEVICE_MAX_MEM_ALLOC_SIZE, device_info->max_mem_alloc_size, verbose );
   get_info( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, device_info->global_mem_cacheline_size, verbose );
   get_info( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, device_info->global_mem_cache_size, verbose );

   get_info( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, device_info->global_mem_cache_type, verbose);
   if (verbose)
   {
      printf("\t%-40s = ", "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE");
      if (device_info->global_mem_cache_type == CL_NONE)
         printf("%s\n", "CL_NONE");
      else if (device_info->global_mem_cache_type == CL_READ_ONLY_CACHE)
         printf("%s\n", "CL_READ_ONLY_CACHE");
      else if (device_info->global_mem_cache_type == CL_READ_WRITE_CACHE)
         printf("%s\n", "CL_READ_WRITE_CACHE");
   }

   get_info( CL_DEVICE_LOCAL_MEM_SIZE, device_info->local_mem_size, verbose );
   get_info( CL_DEVICE_LOCAL_MEM_TYPE, device_info->local_mem_type, verbose );
   if (verbose)
   {
      printf("\t%-40s = %s\n", "CL_DEVICE_LOCAL_MEM_TYPE", (device_info->local_mem_type == CL_LOCAL) ? "LOCAL" : "GLOBAL");
   }

   get_info( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, device_info->max_constant_buffer_size, verbose );
   get_info( CL_DEVICE_MAX_CONSTANT_ARGS, device_info->max_constant_args, verbose );

   get_info( CL_DEVICE_DOUBLE_FP_CONFIG, device_info->fp_config, verbose );

   #undef get_char_info
   #undef get_info
}

typedef struct
{
   platformInfo_t platform_info;
   deviceInfo_t device_info;

   cl_context context;
   cl_command_queue command_queue;
   cl_program program;

   size_t blockSize;
   size_t numBlocks;
   int vectorSize;

   int use_queue;
}
cl_data_t;

static cl_data_t __cl_data;
static cl_data_t *cl_data = NULL;

int cl_init (const ckdata_t *ck)
{
   cl_int ret;

   if (cl_data != NULL)
      return CL_SUCCESS;

   cl_data = &__cl_data;

   getPlatformInfo(&cl_data->platform_info);

   getDeviceInfo(&cl_data->device_info, &cl_data->platform_info);

   cl_data->context = clCreateContext(NULL, 1, &cl_data->device_info.device_id, NULL, NULL, &ret);
   if (ret != CL_SUCCESS )
   {
      fprintf(stderr,"clError: clCreateContext ret=%d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   cl_data->command_queue = clCreateCommandQueue(cl_data->context, cl_data->device_info.device_id, 0, &ret);
   if (ret != CL_SUCCESS )
   {
      fprintf(stderr,"clError: clCreateCommandQueue ret=%d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
      fprintf(stderr,"clError: clCreateCommandQueue ret=%d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   cl_data->use_queue = 1;
   {
      char *env = getenv("QUEUE");
      if (env)
         if (isdigit(*env))
            cl_data->use_queue = (atoi(env) != 0);
   }

   cl_data->blockSize = 1;
   {
      char *env = getenv("BLOCKSIZE");
      if (env)
         if (isdigit(*env))
            cl_data->blockSize = atoi(env);
   }

   cl_data->numBlocks = cl_data->device_info.max_compute_units;//getDeviceMaxComputeUnits( device_id);
   {
      char *env = getenv("NUMBLOCKS");
      if (env)
         if (isdigit(*env))
            cl_data->numBlocks = atoi(env);
   }

   cl_data->vectorSize = 1;
   {
      char *env = getenv("VECTOR");
      if (env && isdigit(*env))
         cl_data->vectorSize = atoi(env);
   }
   assert ( isPower2(cl_data->vectorSize) );
   assert ( isPower2(cl_data->blockSize) );
   //assert ( blockSize >= vectorSize );
   if (cl_data->blockSize <  cl_data->vectorSize)
      cl_data->blockSize = cl_data->vectorSize;

   cl_data->blockSize /= cl_data->vectorSize;

   enum { max_source_size = 0x400000 }; // 4MB
   char *program_source_str = (char*)malloc(sizeof(char)*max_source_size);
   size_t program_source_size = 0;

   if (1)
   {
      const char *dp_header = "#if defined(cl_khr_fp64) \n"
                              "#pragma OPENCL EXTENSION cl_khr_fp64 : enable  \n"
                              "#elif defined(cl_amd_fp64) \n"
                              "#pragma OPENCL EXTENSION cl_amd_fp64 : enable  \n"
                              "#endif \n";

      strcpy (program_source_str, dp_header);
      program_source_size += strlen(dp_header);
   }

   program_source_size += sprintf(program_source_str + program_source_size, "#define __sizeof_ckdata (%d)\n", sizeof(ckdata_t));
   if (__Alignment)
   program_source_size += sprintf(program_source_str + program_source_size, "#define __Alignment     (%d)\n", __Alignment);
   program_source_size += sprintf(program_source_str + program_source_size, "#define __ValueSize      %d\n", cl_data->vectorSize);
   program_source_size += sprintf(program_source_str + program_source_size, "#define __blockSize      %d\n", cl_data->blockSize);

   if (cl_data->use_queue)
      program_source_size += sprintf(program_source_str + program_source_size, "#define __EnableQueue\n");

   // Load the common macros ...
   program_source_size += load_source_from_file ("cl_macros.h", program_source_str + program_source_size, max_source_size - program_source_size);

   // Load the header and source files text ...
   program_source_size += load_source_from_file ("cklib.h", program_source_str + program_source_size, max_source_size - program_source_size);
   program_source_size += load_source_from_file ("cklib.c", program_source_str + program_source_size, max_source_size - program_source_size);

   program_source_size += load_source_from_file ("rk.h", program_source_str + program_source_size, max_source_size - program_source_size);
   program_source_size += load_source_from_file ("rk.c", program_source_str + program_source_size, max_source_size - program_source_size);

   program_source_size += load_source_from_file ("ros.h", program_source_str + program_source_size, max_source_size - program_source_size);
   program_source_size += load_source_from_file ("ros.c", program_source_str + program_source_size, max_source_size - program_source_size);

   //if (1)
   //if (device_type == CL_DEVICE_TYPE_GPU)
   if (cl_data->device_info.type != CL_DEVICE_TYPE_ACCELERATOR && 0)
   {
      printf("adding __constant __ckobj for device = %d\n", cl_data->device_info.type);

      // Build the constant data table ...
      program_source_size += write_ckdata_table (ck, "__ckobj", program_source_str + program_source_size, max_source_size - program_source_size);
   }

   // Load source from file ...
   //program_source_size += load_source_from_file ("cklib.c", program_source_str + program_source_size, max_source_size - program_source_size);
   program_source_size += load_source_from_file ("ck_driver.cl", program_source_str + program_source_size, max_source_size - program_source_size);
   /*{
      FILE *fp = fopen("ck_driver.cl", "rb");
      assert (fp != NULL);

      fseek (fp, 0, SEEK_END);
      size_t sz = ftell(fp);
      rewind (fp);

      size_t space = max_source_size - program_source_size;
      assert (sz < space);

      char *src = program_source_str + program_source_size;
      program_source_size += fread(src, sizeof(char), space, fp);

      fclose(fp);
   }*/

   //printf("%s\n", program_source_str);
   if (1)
   {
      const char *kernel_source_filename = "__kernel_source.cl";
      FILE *fp = fopen(kernel_source_filename,"w");
      if (fp == NULL)
      {
         fprintf(stderr,"Error opening %s\n", kernel_source_filename);
         exit(-1);
      }

      fprintf(fp, "%s\n", program_source_str);

      fclose(fp);
   }

   /* Build Program */
   cl_data->program = clCreateProgramWithSource(cl_data->context, 1, (const char **)&program_source_str, (const size_t *)&program_source_size, &ret);
   assert( ret == CL_SUCCESS );

   char build_options[1024];
   strcpy (build_options, "-I. ");
   //strcat (build_options + strlen(build_options), " -D__arrayStride=1");
// sprintf(build_options + strlen(build_options), " -D__arrayStride=%d", 32);
   if (cl_data->platform_info.is_nvidia)
      strcat (build_options + strlen(build_options), " -cl-nv-verbose");
   printf("build_options = %s\n", build_options);

   ret = clBuildProgram(cl_data->program, 1, &cl_data->device_info.device_id, build_options, NULL, NULL);
   //if (ret != CL_SUCCESS)
   {
      fprintf(stderr,"clBuildProgram = %d\n", ret);

      cl_build_status build_status;
      CL_EXEC( clGetProgramBuildInfo (cl_data->program, cl_data->device_info.device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL) );
      printf("%-40s =\n%d\n", "CL_PROGRAM_BUILD_STATUS", build_status);

      char build_log[4096];
      size_t build_log_size;
      CL_EXEC( clGetProgramBuildInfo (cl_data->program, cl_data->device_info.device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &build_log_size) );
      if (build_log_size > 0)
      {
         build_log[build_log_size+1] = '\0';
         printf("%-40s = %s\n", "CL_PROGRAM_BUILD_LOG", build_log);
      }

      if (ret != CL_SUCCESS) return 1;

      if (cl_data->platform_info.is_nvidia && 0)
      {
         /* Query binary (PTX file) size */
         size_t binary_size;
         CL_EXEC( clGetProgramInfo (cl_data->program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL) );

         /* Read binary (PTX file) to memory buffer */
         unsigned char *ptx_binary = (unsigned char *)malloc(binary_size);
         CL_EXEC( clGetProgramInfo (cl_data->program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptx_binary, NULL) );
         /* Save PTX to add_vectors_ocl.ptx */
         FILE *ptx_binary_file = fopen("ptx_binary_ocl.ptx", "wb");
         fwrite(ptx_binary, sizeof(char), binary_size, ptx_binary_file);
         fclose(ptx_binary_file);

         free(ptx_binary);
      }
   }

   free(program_source_str);

   return CL_SUCCESS;
}
int cl_ck_driver(double p, double T, double *y, ckdata_t *ck, double *udot_ref, int nfe)
{
   cl_init(ck);

   cl_int ret;

   double t_start = WallClock();

   /* Extract the kernel from the program */
   cl_kernel ck_kernel = clCreateKernel(cl_data->program, "ck_driver_vec", &ret);
   if (ret != CL_SUCCESS)
   {
      fprintf(stderr,"clCreateKernel error = %d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   // Query the kernel's info
   if (1)
   {
      kernelInfo_t kernel_info;
      getKernelInfo (&kernel_info, ck_kernel, cl_data->device_info.device_id);
   }

   const int kk = ck->n_species;
   const int neq = kk+1;

   int lenrwk = (ck_lenrwk(ck) + 2*neq)*cl_data->vectorSize;
   printf("ck_lenrwk = %d %d\n", ck_lenrwk(ck), lenrwk);

   size_t numThreads = cl_data->blockSize * cl_data->numBlocks;
   printf("NP = %d, blockSize = %lu, vectorSize = %d numBlocks = %lu, numThreads = %lu\n", nfe, cl_data->blockSize, cl_data->vectorSize, cl_data->numBlocks, numThreads);

   //for (int k = 0; k < neq; ++k) printf("y[%d]=%e\n", k, y[k]);

   double t_data = WallClock();

   cl_mem buffer_y = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*kk, NULL);
   cl_mem buffer_udot = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*nfe, NULL);
   cl_mem buffer_rwk = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*lenrwk*numThreads, NULL);
   cl_mem buffer_ck = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(ckdata_t), NULL);

   {
      size_t _memsize = sizeof(double)*kk
                      + sizeof(double)*neq*nfe
                      + sizeof(double)*lenrwk*numThreads
                      + sizeof(ckdata_t);
      printf("Device memory size = %lu (KB)\n", _memsize);
   }

   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_y, CL_TRUE, 0, sizeof(double)*kk, y, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_ck, CL_TRUE, 0, sizeof(ckdata_t), ck, 0, NULL, NULL) )

   t_data = WallClock() - t_data;
   printf("Host->Dev + alloc = %lu %f (ms)\n", sizeof(double)*kk + sizeof(ckdata_t), 1000*t_data);

   /* Set kernel argument */
   CL_EXEC( clSetKernelArg(ck_kernel, 0, sizeof(double), &p) );
   CL_EXEC( clSetKernelArg(ck_kernel, 1, sizeof(double), &T) );
   CL_EXEC( clSetKernelArg(ck_kernel, 2, sizeof(cl_mem), &buffer_y) );
   CL_EXEC( clSetKernelArg(ck_kernel, 3, sizeof(cl_mem), &buffer_udot) );
   CL_EXEC( clSetKernelArg(ck_kernel, 4, sizeof(cl_mem), &buffer_ck) );
   CL_EXEC( clSetKernelArg(ck_kernel, 5, sizeof(cl_mem), &buffer_rwk) );
   CL_EXEC( clSetKernelArg(ck_kernel, 6, sizeof(int), &nfe) );

   double t0 = WallClock();

   /* Execute kernel */
   cl_event ev;
   ret = clEnqueueNDRangeKernel (cl_data->command_queue, ck_kernel,
                                 1 /* work-group dims */,
                                 NULL /* offset */,
                                 &numThreads /* global work size */,
                                 &cl_data->blockSize /* local work-group size */,
                                 0, NULL, /* wait list */
                                 &ev /* this kernel's event */);
   if (ret != CL_SUCCESS) {
      fprintf(stderr,"Error in clEnqueueTask: ret = %d\n", ret);
      //return 1;
   }

   /* Wait for the kernel to finish */
   clWaitForEvents(1, &ev);

   double t1 = WallClock();
   if (ret == CL_SUCCESS) printf("CL time = %f\n", (t1-t0));

   t_data = WallClock();
   double *h_udot = (double *) malloc(sizeof(double)*neq*nfe);
   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_udot, CL_TRUE, 0, sizeof(double)*neq*nfe, h_udot, 0, NULL, NULL) );
   t_data = WallClock() - t_data;
   if (ret == CL_SUCCESS)
   {
      printf("Dev->Host size = %lu %f (ms)\n", sizeof(double)*neq*nfe, 1000*t_data);

      if (udot_ref)
      {
      for (int k = 0; k < neq; ++k)
      {
         double err = 0, ref = 0;

         #pragma omp parallel for reduction(+:err, ref)
         for (int n = 0; n < nfe; ++n)
         {
            double T0 = T + (1000.*n)/nfe;
            double *udot_ = udot_ref + (neq*n);

            double dif = udot_[k] - h_udot[k + neq*n];
            err += (dif*dif);
            ref += (udot_[k] * udot_[k]);
            //std::cout << "T: " << T << " dif: " << dif << "\n";
            //for (int k = 0; k < neq; ++k)
            //   printf("k=%d %e %e\n", k, udot[k], h_udot[k+(neq)*n]);
            //printf("n=%d %e %e\n", n, udot_[idx], h_udot[idx+(neq)*n]);
            //printf("%d: udot[%d]=%e, %e\n", n, k, udot_[k], h_udot[k+(neq)*n]);
         }

         err = sqrt(err) /  nfe;
         ref = sqrt(ref) /  nfe;

         //printf("Ref time = %f\n", t1-t0);
         if (ref > sqrt(DBL_EPSILON))
         {
            if (ref < 1e-20) ref = 1;
            printf("Rel error[%d] = %e\n", k, err/ref);
         }
      }
      }
   }

   clReleaseKernel (ck_kernel);
   clReleaseMemObject (buffer_y);
   clReleaseMemObject (buffer_udot);
   clReleaseMemObject (buffer_ck);
   clReleaseMemObject (buffer_rwk);

   free(h_udot);

   printf("Total CL time = %f\n", WallClock() - t_start);

   return 0;
}
int cl_ck_driver_array(double *p, double *T, double *y, ckdata_t *ck, double *udot_ref, int nfe)
{
   cl_init(ck);

   cl_int ret;

   double t_start = WallClock();

   /* Extract the kernel from the program */
   cl_kernel ck_kernel = clCreateKernel(cl_data->program, "ck_driver_vec_array", &ret);
   if (ret != CL_SUCCESS)
   {
      fprintf(stderr,"clCreateKernel error = %d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   // Query the kernel's info
   if (1)
   {
      kernelInfo_t kernel_info;
      getKernelInfo (&kernel_info, ck_kernel, cl_data->device_info.device_id);
   }

   const int kk = ck->n_species;
   const int neq = kk+1;

   int lenrwk = (ck_lenrwk(ck) + 2*neq)*cl_data->vectorSize;
   printf("ck_lenrwk = %d %d\n", ck_lenrwk(ck), lenrwk);

   size_t numThreads = cl_data->blockSize * cl_data->numBlocks;
   printf("NP = %d, blockSize = %lu, vectorSize = %d numBlocks = %lu, numThreads = %lu\n", nfe, cl_data->blockSize, cl_data->vectorSize, cl_data->numBlocks, numThreads);

   double t_data = WallClock();

   cl_mem buffer_p = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*nfe, NULL);
   cl_mem buffer_T = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*nfe, NULL);
   cl_mem buffer_y = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*kk*nfe, NULL);
   cl_mem buffer_udot = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*nfe, NULL);
   cl_mem buffer_rwk = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*lenrwk*numThreads, NULL);
   cl_mem buffer_ck = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(ckdata_t), NULL);

   {
      size_t _memsize = sizeof(double)*kk*nfe + sizeof(double)*nfe*2
                      + sizeof(double)*neq*nfe
                      + sizeof(double)*lenrwk*numThreads
                      + sizeof(ckdata_t);
      printf("Device memory size = %lu (KB)\n", _memsize);
   }

   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_p, CL_TRUE, 0, sizeof(double)*nfe, p, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_T, CL_TRUE, 0, sizeof(double)*nfe, T, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_y, CL_TRUE, 0, sizeof(double)*kk*nfe, y, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_ck, CL_TRUE, 0, sizeof(ckdata_t), ck, 0, NULL, NULL) )

   t_data = WallClock() - t_data;
   printf("Host->Dev + alloc = %lu %f (ms)\n", sizeof(double)*kk + sizeof(ckdata_t), 1000*t_data);

   /* Set kernel argument */
   CL_EXEC( clSetKernelArg(ck_kernel, 0, sizeof(cl_mem), &buffer_p) );
   CL_EXEC( clSetKernelArg(ck_kernel, 1, sizeof(cl_mem), &buffer_T) );
   CL_EXEC( clSetKernelArg(ck_kernel, 2, sizeof(cl_mem), &buffer_y) );
   CL_EXEC( clSetKernelArg(ck_kernel, 3, sizeof(cl_mem), &buffer_udot) );
   CL_EXEC( clSetKernelArg(ck_kernel, 4, sizeof(cl_mem), &buffer_ck) );
   CL_EXEC( clSetKernelArg(ck_kernel, 5, sizeof(cl_mem), &buffer_rwk) );
   CL_EXEC( clSetKernelArg(ck_kernel, 6, sizeof(int), &nfe) );

   double t0 = WallClock();

   /* Execute kernel */
   cl_event ev;
   ret = clEnqueueNDRangeKernel (cl_data->command_queue, ck_kernel,
                                 1 /* work-group dims */,
                                 NULL /* offset */,
                                 &numThreads /* global work size */,
                                 &cl_data->blockSize /* local work-group size */,
                                 0, NULL, /* wait list */
                                 &ev /* this kernel's event */);
   if (ret != CL_SUCCESS) {
      fprintf(stderr,"Error in clEnqueueTask: ret = %d\n", ret);
      //return 1;
   }

   /* Wait for the kernel to finish */
   clWaitForEvents(1, &ev);

   double t1 = WallClock();
   if (ret == CL_SUCCESS) printf("CL time = %f\n", (t1-t0));

   t_data = WallClock();
   double *h_udot = (double *) malloc(sizeof(double)*neq*nfe);
   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_udot, CL_TRUE, 0, sizeof(double)*neq*nfe, h_udot, 0, NULL, NULL) );
   t_data = WallClock() - t_data;
   if (ret == CL_SUCCESS)
   {
      printf("Dev->Host size = %lu %f (ms)\n", sizeof(double)*neq*nfe, 1000*t_data);

      if (udot_ref)
      {
      for (int k = 0; k < neq; ++k)
      {
         double err = 0, ref = 0;

         #pragma omp parallel for reduction(+:err, ref)
         for (int n = 0; n < nfe; ++n)
         {
            double *udot_ = udot_ref + (neq*n);

            double dif = udot_[k] - h_udot[k + neq*n];
            err += (dif*dif);
            ref += (udot_[k] * udot_[k]);
            //for (int k = 0; k < neq; ++k)
            //   printf("k=%d %e %e\n", k, udot[k], h_udot[k+(neq)*n]);
            //printf("n=%d %e %e\n", n, udot_[idx], h_udot[idx+(neq)*n]);
            //printf("%d: udot[%d]=%e, %e\n", n, k, udot_[k], h_udot[k+(neq)*n]);
         }

         err = sqrt(err) /  nfe;
         ref = sqrt(ref) /  nfe;

         //printf("Ref time = %f\n", t1-t0);
         if (ref > sqrt(DBL_EPSILON))
         {
            if (ref < 1e-20) ref = 1;
            printf("Rel error[%d] = %e\n", k, err/ref);
         }
      }
      }
   }

   clReleaseKernel (ck_kernel);
   clReleaseMemObject (buffer_p);
   clReleaseMemObject (buffer_T);
   clReleaseMemObject (buffer_y);
   clReleaseMemObject (buffer_udot);
   clReleaseMemObject (buffer_ck);
   clReleaseMemObject (buffer_rwk);

   free(h_udot);

   printf("Total CL time = %f\n", WallClock() - t_start);

   return 0;
}
int cl_rk_driver(double p, double *u_in, double *u_out_ref, ckdata_t *ck, rk_t *rk, int numProblems)
{
   cl_init(ck);

   cl_int ret;

   const int kk = ck->n_species;
   const int neq = kk+1;

   size_t numThreads = cl_data->blockSize * cl_data->numBlocks;
   printf("NP = %d, blockSize = %lu, vectorSize = %d numBlocks = %lu, numThreads = %lu\n", numProblems, cl_data->blockSize, cl_data->vectorSize, cl_data->numBlocks, numThreads);

   int lenrwk = (ck_lenrwk(ck) + rk_lenrwk(rk) + neq) * cl_data->vectorSize;
   printf("ck_lenrwk = %d rk_lenrwk = %d %d\n", ck_lenrwk(ck), rk_lenrwk(rk), lenrwk);

   cl_kernel rk_kernel;
   int use_queue = cl_data->use_queue;
   if (cl_data->vectorSize > 1)
      if (use_queue)
         rk_kernel = clCreateKernel(cl_data->program, "rk_driver_vec_queue", &ret);
      else
         rk_kernel = clCreateKernel(cl_data->program, "rk_driver_vec", &ret);
   else
   {
      if (use_queue)
         rk_kernel = clCreateKernel(cl_data->program, "rk_driver_queue", &ret);
      else
         rk_kernel = clCreateKernel(cl_data->program, "rk_driver", &ret);
   }
   if (ret != CL_SUCCESS)
   {
      fprintf(stderr,"clCreateKernel error = %d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   {
      kernelInfo_t kernel_info;
      getKernelInfo (&kernel_info, rk_kernel, cl_data->device_info.device_id);
   }

   cl_mem buffer_uin = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*numProblems, NULL);
   cl_mem buffer_uout = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*numProblems, NULL);
   cl_mem buffer_ck = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(ckdata_t), NULL);
   cl_mem buffer_rk = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(rk_t), NULL);
   cl_mem buffer_rwk = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*lenrwk*numThreads, NULL);
   cl_mem buffer_counters = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(rk_counters_t)*numProblems, NULL);

   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_uin, CL_TRUE, 0, sizeof(double)*neq*numProblems, u_in, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_ck, CL_TRUE, 0, sizeof(ckdata_t), ck, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_rk, CL_TRUE, 0, sizeof(rk_t), rk, 0, NULL, NULL) )

   cl_mem buffer_queue = NULL;
   if (use_queue)
   {
      buffer_queue = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(int), NULL);
      int queue_val = 0;
      CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_queue, CL_TRUE, 0, sizeof(int), &queue_val, 0, NULL, NULL) )
      printf("Queue enabled\n");
   }

   /* Set kernel argument */
   int argc = 0;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(double), &p) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_uin) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_uout) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_ck) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_rk) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_rwk) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_counters) ); argc++;
   CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(int), &numProblems) ); argc++;
   if (use_queue)
   {
      CL_EXEC( clSetKernelArg(rk_kernel, argc, sizeof(cl_mem), &buffer_queue) ); argc++;
   }

   double _t0 = WallClock();

   /* Execute kernel */
   cl_event ev;
   ret = clEnqueueNDRangeKernel (cl_data->command_queue, rk_kernel,
                                 1 /* work-group dims */,
                                 NULL /* offset */,
                                 &numThreads /* global work size */,
                                 &cl_data->blockSize /* local work-group size */,
                                 0, NULL, /* wait list */
                                 &ev /* this kernel's event */);
   if (ret != CL_SUCCESS) {
      fprintf(stderr,"Error in clEnqueueTask: ret = %d\n", ret);
      //return 1;
   }

   /* Wait for the kernel to finish */
   clWaitForEvents(1, &ev);

   double _t1 = WallClock();
   printf("RK time = %f\n", (_t1-_t0));

   double *u_out = (double *) malloc(sizeof(double)*neq*numProblems);

   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_uout, CL_TRUE, 0, sizeof(double)*neq*numProblems, u_out, 0, NULL, NULL) );

   rk_counters_t *counters = (rk_counters_t *) malloc(sizeof(rk_counters_t)*numProblems);
   if (counters == NULL)
   {
      fprintf(stderr,"Allocation error %s %d\n", __FILE__, __LINE__);
      exit(-1);
   } 
   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_counters, CL_TRUE, 0, sizeof(rk_counters_t)*numProblems, counters, 0, NULL, NULL) );

   //if (numProblems < 16)
   {
      int stride = numProblems / 16;
      if (stride == 0) stride = 1;
      for (int n = 0; n < numProblems; /*n++*/ n += stride)
      {
         //printf("u[%d] = %f %d %d %f\n", n, u_out[n*neq+kk], counters[n].nsteps, counters[n].niters, u_out_ref[n*neq+kk]);
         printf("u[%d] = %e %d %d %f %f %e\n", n, u_out[n*neq+kk], counters[n].nsteps, counters[n].niters, u_in[n*neq+kk], u_out_ref[n*neq+kk], u_in[n*neq+kk] - u_out[n*neq+kk]);
      }
   }

   int nst_ = 0, nit_ = 0;
   for (int i = 0; i < numProblems; ++i)
   {
      nst_ += counters[i].nsteps;
      nit_ += counters[i].niters;
   }
   printf("nst = %d, nit = %d\n", nst_, nit_);

   if (u_out_ref != NULL)
   {
      for (int k = 0; k < neq; ++k)
      {
         double err = 0, ref = 0;
         for (int i = 0; i < numProblems; ++i)
         {
            double diff = u_out[i*neq+k] - u_out_ref[i*neq+k];
            err += (diff*diff);
            ref += u_out_ref[i*neq+k];
         }
         err = sqrt(err) / numProblems;
         ref = sqrt(ref) / numProblems;
         if (ref > sqrt(DBL_EPSILON))
         {
            if (ref < 1e-20) ref = 1;
            printf("Rel error[%d] = %e\n", k, err/ref);
         }
      }
   }

   clReleaseKernel(rk_kernel);
   clReleaseMemObject (buffer_rk);
   clReleaseMemObject (buffer_uin);
   clReleaseMemObject (buffer_uout);
   clReleaseMemObject (buffer_ck);
   clReleaseMemObject (buffer_rwk);

   free(counters);
   free(u_out);

   return 0;
}
int cl_ros_driver(double p, double *u_in, double *u_out_ref, ckdata_t *ck, ros_t *ros, int numProblems)
{
   cl_init(ck);

   double t_start = WallClock();

   cl_int ret;

   const int kk = ck->n_species;
   const int neq = kk+1;

   size_t numThreads = cl_data->blockSize * cl_data->numBlocks;
   printf("NP = %d, blockSize = %lu, vectorSize = %d numBlocks = %lu, numThreads = %lu\n", numProblems, cl_data->blockSize, cl_data->vectorSize, cl_data->numBlocks, numThreads);

   int lenrwk = (ck_lenrwk(ck) + ros_lenrwk(ros) + neq) * cl_data->vectorSize;
   int leniwk = ros_leniwk(ros) * cl_data->vectorSize;
   printf("ck_lenrwk = %d ros_lenrwk = %d ros_leniwk = %d %d %d\n", ck_lenrwk(ck), ros_lenrwk(ros), ros_leniwk(ros), lenrwk, leniwk);

   int use_queue = cl_data->use_queue;
   cl_kernel ros_kernel;
   if (cl_data->vectorSize > 1)
      if (use_queue)
         ros_kernel = clCreateKernel(cl_data->program, "ros_driver_vec_queue", &ret);
      else
         ros_kernel = clCreateKernel(cl_data->program, "ros_driver_vec", &ret);
   else
   {
      if (use_queue)
         ros_kernel = clCreateKernel(cl_data->program, "ros_driver_queue", &ret);
      else
         ros_kernel = clCreateKernel(cl_data->program, "ros_driver", &ret);
   }
   if (ret != CL_SUCCESS)
   {
      fprintf(stderr,"clCreateKernel error = %d %s %d\n", ret, __FILE__, __LINE__);
      exit(-1);
   }

   {
      kernelInfo_t kernel_info;
      getKernelInfo (&kernel_info, ros_kernel, cl_data->device_info.device_id);
   }

   double t_data = WallClock();

   cl_mem buffer_uin = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*numProblems, NULL);
   cl_mem buffer_uout = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*neq*numProblems, NULL);
   cl_mem buffer_ck = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(ckdata_t), NULL);
   cl_mem buffer_ros = CreateBuffer (&cl_data->context, CL_MEM_READ_ONLY, sizeof(ros_t), NULL);
   cl_mem buffer_iwk = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(int)*leniwk*numThreads, NULL);
   cl_mem buffer_rwk = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(double)*lenrwk*numThreads, NULL);
   cl_mem buffer_counters = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(ros_counters_t)*numProblems, NULL);

   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_uin, CL_TRUE, 0, sizeof(double)*neq*numProblems, u_in, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_ck, CL_TRUE, 0, sizeof(ckdata_t), ck, 0, NULL, NULL) )
   CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_ros, CL_TRUE, 0, sizeof(ros_t), ros, 0, NULL, NULL) )

   cl_mem buffer_queue = NULL;
   if (use_queue)
   {
      buffer_queue = CreateBuffer (&cl_data->context, CL_MEM_READ_WRITE, sizeof(int), NULL);
      int queue_val = 0;
      CL_EXEC( clEnqueueWriteBuffer(cl_data->command_queue, buffer_queue, CL_TRUE, 0, sizeof(int), &queue_val, 0, NULL, NULL) )
      printf("Queue enabled\n");
   }

   printf("Host->dev %f (ms)\n", 1000*(WallClock() - t_data));

   /* Set kernel argument */
   int argc = 0;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(double), &p) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_uin) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_uout) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_ck) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_ros) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_iwk) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_rwk) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_counters) ); argc++;
   CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(int), &numProblems) ); argc++;
   if (use_queue)
   {
      CL_EXEC( clSetKernelArg(ros_kernel, argc, sizeof(cl_mem), &buffer_queue) ); argc++;
   }

   double _t0 = WallClock();

   /* Execute kernel */
   cl_event ev;
   ret = clEnqueueNDRangeKernel (cl_data->command_queue, ros_kernel,
                                 1 /* work-group dims */,
                                 NULL /* offset */,
                                 &numThreads /* global work size */,
                                 &cl_data->blockSize /* local work-group size */,
                                 0, NULL, /* wait list */
                                 &ev /* this kernel's event */);
   if (ret != CL_SUCCESS) {
      fprintf(stderr,"Error in clEnqueueTask: ret = %d\n", ret);
      //return 1;
   }

   /* Wait for the kernel to finish */
   clWaitForEvents(1, &ev);

   double _t1 = WallClock();
   printf("ROS time = %f\n", (_t1-_t0));

   double *u_out = (double *) malloc(sizeof(double)*neq*numProblems);

   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_uout, CL_TRUE, 0, sizeof(double)*neq*numProblems, u_out, 0, NULL, NULL) );

   ros_counters_t *counters = (ros_counters_t *) malloc(sizeof(ros_counters_t)*numProblems);
   if (counters == NULL)
   {
      fprintf(stderr,"Allocation error %s %d\n", __FILE__, __LINE__);
      exit(-1);
   } 
   CL_EXEC( clEnqueueReadBuffer(cl_data->command_queue, buffer_counters, CL_TRUE, 0, sizeof(ros_counters_t)*numProblems, counters, 0, NULL, NULL) );

   printf("Dev->host %f (ms)\n", 1000*(WallClock() - _t1));

   if (1)//numProblems < 16)
   {
      int stride = numProblems / 16; // + ((numProblems % 16 == 0) ? 0 : 1);
      if (stride == 0) stride = 1;
      for (int n = 0; n < numProblems; n += stride)
      //for (int n = 0; n < (numProblems > 16 ? 16 : numProblems); n++)
      {
         printf("u[%d] = %f %d %d %f %f %e\n", n, u_out[n*neq+kk], counters[n].nst, counters[n].niters, u_in[n*neq+kk], u_out_ref[n*neq+kk], u_in[n*neq+kk] - u_out[n*neq+kk]);
      }
   }

   int nst_ = 0, nit_ = 0;
   for (int i = 0; i < numProblems; ++i)
   {
      nst_ += counters[i].nst;
      nit_ += counters[i].niters;
   }
   printf("nst = %d, nit = %d\n", nst_, nit_);

   if (u_out_ref != NULL)
   {
      for (int k = 0; k < neq; ++k)
      {
         double err = 0, ref = 0;
         for (int i = 0; i < numProblems; ++i)
         {
            double diff = u_out[i*neq+k] - u_out_ref[i*neq+k];
            err += (diff*diff);
            ref += u_out_ref[i*neq+k];
         }
         err = sqrt(err) / numProblems;
         ref = sqrt(ref) / numProblems;
         if (ref > sqrt(DBL_EPSILON))
         {
            if (ref < 1e-20) ref = 1;
            printf("Rel error[%d] = %e\n", k, err/ref);
         }
      }
   }

   clReleaseKernel(ros_kernel);
   clReleaseMemObject (buffer_ros);
   clReleaseMemObject (buffer_uin);
   clReleaseMemObject (buffer_uout);
   clReleaseMemObject (buffer_ck);
   clReleaseMemObject (buffer_iwk);
   clReleaseMemObject (buffer_rwk);
   if (use_queue)
      clReleaseMemObject (buffer_queue);

   free(counters);
   free(u_out);

   printf("Total CL time = %f\n", WallClock() - t_start);

   return 0;
}
