#ifndef __Vector_h
#define __Vector_h

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#ifdef __ALIGNMENT
#define __default_alignment __ALIGNMENT
#endif

#ifndef __default_alignment
#if defined(__MIC__) || defined(_OPENMP)
   #warning 'defining alignment = 64 for MIC || OpenMP'
   #define __default_alignment (64)
#endif
#endif

#ifndef __default_alignment
   //#warning 'defining default alignment = sizeof(double) for HOST'
   //#define __default_alignment (sizeof(double))
   #warning 'defining default alignment = 16'
   #define __default_alignment (16)
#endif

template <typename ValueType, int _Alignment = __default_alignment>
struct VectorType
{
   enum { alignment = _Alignment };

   typedef ValueType value_type;

   bool is_ref;
   value_type *ptr;
   int len;

   VectorType (void) : len(0), ptr(NULL), is_ref(false) {}
   explicit VectorType (const int len) : len(len), ptr(NULL), is_ref(false)
   {
      //ptr = new value_type [len];
      //if (ptr == NULL) {
      //   fprintf(stderr,"Allocation error %s %d\n", __FILE__, __LINE__);
      //   exit(1);
      //}
      this->resize(this->len);
   }
   explicit VectorType (int len, value_type *ptr) : len(len), ptr(ptr), is_ref(true) {}
   explicit VectorType (value_type *ptr) : len(INT_MAX), ptr(ptr), is_ref(true) {}
   explicit VectorType (const VectorType& x) : len(x.len), ptr(x.ptr), is_ref(true) {}

   ~VectorType()
   {
      //if (this->is_ref == false and this->ptr)
      //   delete [] this->ptr;
      //if (this->is_ref == false and this->ptr)
      //   resize(0);
      if (not(this->is_ref))
         this->clear();
   }

   void clear (void)
   {
      assert (not(this->is_ref));

      if (this->is_ref == false and this->ptr)
         if (alignment)
            free(this->ptr);
         else
            delete [] this->ptr;

      this->ptr = NULL;
      this->len = 0;
   }

   void resize (const int n)
   {
      assert (not(this->is_ref));
      //printf("inside VectorType::resize %d %d %x\n", n, len, ptr);

      if (n == 0)
      {
         //if (this->ptr)
         //   delete [] this->ptr;
         //this->ptr = NULL;
         this->clear();
      }
      else
      {
         value_type *p = NULL;

         if (alignment)
         {
            int ierr = posix_memalign((void**)&p, (size_t)alignment, sizeof(value_type)*n);
            if (ierr) {
               fprintf(stderr,"Aligned allocation error %s %d\n", __FILE__, __LINE__);
               exit(1);
            }
         }
         else
         {
            p = new value_type [n];
            if (p == NULL) {
               fprintf(stderr,"Allocation error %s %d\n", __FILE__, __LINE__);
               exit(1);
            }
         }

         if (this->len and this->ptr)
         {
            size_t ncopy = std::min(n,this->len);
            std::copy(this->ptr, this->ptr + ncopy, p);
            //delete [] this->ptr;
            this->clear();
         }

         this->ptr = p;
      }

      this->len = n;

      //printf("leaving VectorType::resize %d %d %x\n", n, len, ptr);
   }

   //const value_type & operator[] (const int i) const { return this->ptr[i]; }
   //      value_type & operator[] (const int i)       { return this->ptr[i]; }
   inline const value_type & operator[] (const int i) const { return *(this->ptr + i); }
   inline       value_type & operator[] (const int i)       { return *(this->ptr + i); }

   void operator += (const size_t offset) { this->ptr += offset; }
   void operator -= (const size_t offset) { this->ptr -= offset; }

   inline       value_type* getPointer (void)       { return this->ptr; }
   inline const value_type* getPointer (void) const { return this->ptr; }

   inline       value_type* getPointer (const int i)       { return this->ptr + i; }
   inline const value_type* getPointer (const int i) const { return this->ptr + i; }

   int size(void) const { return (len); }
   //value_type* begin(void) { return (ptr); }
   //value_type* end(void) { return (ptr + len); }
};

#endif
