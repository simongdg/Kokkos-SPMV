#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#ifdef KOKKOS_USE_CUSPARSE
#include <cusparse.h>
#endif
#include <Kokkos_Core.hpp>
#include <Kokkos_Blas1_MV.hpp>
#include <Kokkos_MV.hpp>
#include <Kokkos_Sparse.hpp>
#include <impl/Kokkos_Timer.hpp>


typedef Kokkos::DefaultExecutionSpace execution_space;

int rows_per_thread; int team_size; int vector_length; int idx_offset;
#ifdef INT64
typedef long long int LocalOrdinalType;
#else
typedef int LocalOrdinalType;
#endif
template< typename ScalarType , typename OrdinalType>
int SparseMatrix_MatrixMarket_read(const char* filename, OrdinalType &nrows, OrdinalType &ncols, OrdinalType &nnz, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  FILE* file = fopen(filename,"r");
  char line[512];
  line[0]='%';
  int count=-1;
  char* symmetric = NULL;
  int nlines;

  while(line[0]=='%')
  {
          fgets(line,511,file);
          count++;
          if(count==0) symmetric=strstr(line,"symmetric");
  }
  rewind(file);
  for(int i=0;i<count;i++)
          fgets(line,511,file);
  fscanf(file,"%i",&nrows);
  fscanf(file,"%i",&ncols);
  fscanf(file,"%i",&nlines);
  printf("Matrix dimension: %i %i %i %s\n",nrows,ncols,nlines,symmetric?"Symmetric":"General");

  if(symmetric) nnz=nlines*2;
  else nnz=nlines;

  OrdinalType* colIndtmp = new OrdinalType[nnz];
  OrdinalType* rowIndtmp = new OrdinalType[nnz];
  double* valuestmp = new double[nnz];
  OrdinalType* priorEntrySameRowInd = new OrdinalType[nnz];
  OrdinalType* lastEntryWithRowInd = new OrdinalType[nrows];
  for(int i=0;i<nrows;i++) lastEntryWithRowInd[i]=-1;
  nnz=0;
  for(int ii=0;ii<nlines;ii++)
  {
          
          fscanf(file,"%i %i %le",&rowIndtmp[nnz],&colIndtmp[nnz],&valuestmp[nnz]);
          if(ii<10||ii>nlines-10) 
            printf("Read: %i %i %i %le\n",nnz,rowIndtmp[nnz],colIndtmp[nnz],valuestmp[nnz]);
          rowIndtmp[nnz]-= idx_offset;
          colIndtmp[nnz]-= idx_offset;
          priorEntrySameRowInd[nnz] = lastEntryWithRowInd[rowIndtmp[nnz]-1];
          lastEntryWithRowInd[rowIndtmp[nnz]-1]=nnz;
          if((symmetric) && (rowIndtmp[nnz]!=colIndtmp[nnz]))
          {
            nnz++;
            rowIndtmp[nnz]=colIndtmp[nnz-1];
            colIndtmp[nnz]=rowIndtmp[nnz-1];
            valuestmp[nnz]=valuestmp[nnz-1];
            priorEntrySameRowInd[nnz] = lastEntryWithRowInd[rowIndtmp[nnz]-1];
            lastEntryWithRowInd[rowIndtmp[nnz]-1]=nnz;
          }
          
          nnz++;
  }

  values = new ScalarType[nnz];
  colInd = new OrdinalType[nnz];
  rowPtr = new OrdinalType[nrows+1];

  int pos = 0;
  for(int row=0;row<nrows;row++)
  {
        int j = lastEntryWithRowInd[row];
        rowPtr[row]=pos;
    while(j>-1)
    {
        values[pos] = valuestmp[j];
        colInd[pos] = colIndtmp[j]-1;
        j = priorEntrySameRowInd[j];
        pos++;
    }
  }
  rowPtr[nrows]=pos;

  printf("Number of Non-Zeros: %i\n",pos);
  delete [] valuestmp;
  delete [] colIndtmp;
  delete [] rowIndtmp;
  delete [] priorEntrySameRowInd;
  delete [] lastEntryWithRowInd;

  size_t min_span = nrows+1;
  size_t max_span = 0;
  size_t ave_span = 0;
  for(int row=0; row<nrows;row++) {
    int min = nrows+1; int max = 0;
    for(int i=rowPtr[row]; i<rowPtr[row+1]; i++) {
      if(colInd[i]<min) min = colInd[i];
      if(colInd[i]>max) max = colInd[i];
    }
    int span = max-min;
    if(span<min_span) min_span = span;
    if(span>max_span) max_span = span;
    ave_span += span;  
  }

  printf("Spans: %lu %lu %lu\n",min_span,max_span,ave_span/nrows);
  return nnz;
}

template< typename ScalarType , typename OrdinalType>
int SparseMatrix_ExtractBinaryGraph(const char* filename, OrdinalType &nrows, OrdinalType &ncols, OrdinalType &nnz, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{

  printf("Extracting Binary Graph... \n");

  nnz = SparseMatrix_MatrixMarket_read<ScalarType,OrdinalType>(filename,nrows,ncols,nnz,values,rowPtr,colInd);

  char * filename_row = new char[strlen(filename)+5];
  char * filename_col = new char[strlen(filename)+5];
  strcpy(filename_row,filename);
  strcpy(filename_col,filename);
  strcat(filename_row,"_row");
  strcat(filename_col,"_col");

  FILE* RowFile = fopen(filename_row,"w");
  FILE* ColFile = fopen(filename_col,"w");

  fwrite ( rowPtr, sizeof(OrdinalType), nrows+1, RowFile);
  fwrite ( colInd, sizeof(OrdinalType), nnz, ColFile);

  size_t min_span = nrows+1;
  size_t max_span = 0;
  size_t ave_span = 0;
  for(int row=0; row<nrows;row++) {
    int min = nrows+1; int max = 0;
    for(int i=rowPtr[row]; i<rowPtr[row+1]; i++) {
      if(colInd[i]<min) min = colInd[i];
      if(colInd[i]>max) max = colInd[i];
    }
    int span = max-min;
    if(span<min_span) min_span = span;
    if(span>max_span) max_span = span;
    ave_span += span;
  }
  printf("Spans: %lu %lu %lu\n",min_span,max_span,ave_span/nrows);

  fclose(RowFile);
  fclose(ColFile);

  printf("Extraction has finished \n");
  return nnz;
}

template< typename ScalarType , typename OrdinalType>
int SparseMatrix_ReadBinaryGraph(const char* filename, OrdinalType &nrows, OrdinalType &ncols, OrdinalType &nnz, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  char * filename_descr = new char[strlen(filename)+7];
  strcpy(filename_descr,filename);
  strcat(filename_descr,"_descr");
  FILE* file = fopen(filename_descr,"r");
  char line[512];
  line[0]='%';
  int count=-1;
  char* symmetric = NULL;
  int nlines;

  while(line[0]=='%')
  {
          fgets(line,511,file);
          count++;
          if(count==0) symmetric=strstr(line,"symmetric");
  }
  rewind(file);
  for(int i=0;i<count;i++)
          fgets(line,511,file);
  fscanf(file,"%i",&nrows);
  fscanf(file,"%i",&ncols);
  fscanf(file,"%i",&nlines);
  printf("Matrix dimension: %i %i %i %s\n",nrows,ncols,nlines,symmetric?"Symmetric":"General");
  if(symmetric) nnz=nlines*2;
  else nnz=nlines;
  fclose(file);

  char * filename_row = new char[strlen(filename)+5];
  char * filename_col = new char[strlen(filename)+5];
  strcpy(filename_row,filename);
  strcpy(filename_col,filename);
  strcat(filename_row,"_row");
  strcat(filename_col,"_col");
  FILE* RowFile = fopen(filename_row,"r");
  FILE* ColFile = fopen(filename_col,"r");

  values = new ScalarType[nnz];
  rowPtr = new OrdinalType[nrows+1];
  colInd = new OrdinalType[nnz];

  fread ( rowPtr, sizeof(OrdinalType), nrows+1, RowFile);
  fread ( colInd, sizeof(OrdinalType), nnz, ColFile);
  fclose(RowFile);
  fclose(ColFile);
    size_t min_span = nrows+1;
  size_t max_span = 0;
  size_t ave_span = 0;
  for(int row=0; row<nrows;row++) {
    int min = nrows+1; int max = 0;
    for(int i=rowPtr[row]; i<rowPtr[row+1]; i++) {
      if(colInd[i]<min) min = colInd[i];
      if(colInd[i]>max) max = colInd[i];
    }
    int span = max-min;
    if(span<min_span) min_span = span;
    if(span>max_span) max_span = span;
    ave_span += span;
  }
  printf("Spans: %lu %lu %lu\n",min_span,max_span,ave_span/nrows);


  return nnz;
}

template< typename ScalarType , typename OrdinalType>
int SparseMatrix_generate(OrdinalType nrows, OrdinalType ncols, OrdinalType &nnz, OrdinalType varianz_nel_row, OrdinalType width_row, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  rowPtr = new OrdinalType[nrows+1];

  OrdinalType elements_per_row = nnz/nrows;
  srand(13721);
  rowPtr[0] = 0;
  for(int row=0;row<nrows;row++)
  {
    int varianz = (1.0*rand()/INT_MAX-0.5)*varianz_nel_row;
    rowPtr[row+1] = rowPtr[row] + elements_per_row+varianz;
  }
  nnz = rowPtr[nrows];
  values = new ScalarType[nnz];
  colInd = new OrdinalType[nnz];
  for(int row=0;row<nrows;row++)
  {
         for(int k=rowPtr[row];k<rowPtr[row+1];k++)
         {
                int pos = (1.0*rand()/INT_MAX-0.5)*width_row+row;
                if(pos<0) pos+=ncols;
                if(pos>=ncols) pos-=ncols;
                colInd[k]= pos;
                values[k] = 100.0*rand()/INT_MAX-50.0;
         }
  }
  return nnz;
}

/*//void MiniFE_MatVec(const int nrows, const int* const Arowoffsets, const int* const Acols, const double* const Acoefs, const double* const xcoefs __attribute__((aligned(64))), double* const ycoefs __attribute__((aligned(64)))) {
void matvec_minife(const int nrows, const int* const Arowoffsets, const int* const Acols, const double* const Acoefs, const double* const xcoefs , double* const ycoefs ) {
#ifdef KOKKOS_HAVE_OPENMP
  #pragma omp parallel for
  for(int row = 0; row < nrows; ++row) {
    const int row_start = Arowoffsets[row];
    const int row_end   = Arowoffsets[row+1];

    double sum = 0;

    #pragma loop_count(15)
    #pragma vector nontemporal
    for(int i = row_start; i < row_end; ++i) {
      sum += Acoefs[i] * xcoefs[Acols[i]];
    }

    ycoefs[row] = sum;
  }
#endif
}*/

template<class AMatrix,
         class XVector,
         class YVector,
         int doalpha,
         int dobeta,
         bool conjugate,
         typename SizeType>
struct SPMV_Functor {
  typedef typename AMatrix::execution_space            execution_space;
  typedef typename AMatrix::non_const_ordinal_type     ordinal_type;
  typedef typename AMatrix::non_const_value_type       value_type;
  typedef SizeType                                     size_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type            team_member;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  const value_type alpha;
  AMatrix  m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_team;

  SPMV_Functor (const value_type alpha_,
               const AMatrix m_A_,
               const XVector m_x_,
               const value_type beta_,
               const YVector m_y_,
               const int rows_per_team_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_team (rows_per_team_)
  {
    static_assert (static_cast<int> (XVector::rank) == 1,
                   "XVector must be a rank 1 View.");
    static_assert (static_cast<int> (YVector::rank) == 1,
                   "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION void
  operator() (const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev,0,rows_per_team), [=] (const ordinal_type& loop) {

      const ordinal_type iRow = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
      if (iRow >= m_A.numRows ()) {
        return;
      }
      const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
      const ordinal_type row_length = static_cast<ordinal_type> (row.length);
      value_type sum = 0;
      
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(dev,row_length), [=] (const ordinal_type& iEntry, value_type& lsum) {
        const value_type val = conjugate ?
                ATV::conj (row.value(iEntry)) :
                row.value(iEntry);
        lsum += val * m_x(row.colidx(iEntry));
      },sum);
   
      Kokkos::single(Kokkos::PerThread(dev), [&] () {
        if (doalpha == -1) {
          sum *= value_type(-1);
        } else if (doalpha * doalpha != 1) {
          sum *= alpha;
        }

        if (dobeta == 0) {
          m_y(iRow) = sum ;
        } else if (dobeta == 1) {
          m_y(iRow) += sum ;
        } else if (dobeta == -1) {
          m_y(iRow) = -m_y(iRow) +  sum;
        } else {
          m_y(iRow) = beta * m_y(iRow) + sum;
        }
      });
    });
  }

  KOKKOS_INLINE_FUNCTION void
  operator() (const team_member& dev,int a) const
  {

    const int rows_per_thread = rows_per_team/dev.team_size();
    // This should be a thread loop as soon as we can use C++11
    for (ordinal_type loop = 0; loop < rows_per_thread; ++loop) {
      // iRow represents a row of the matrix, so its correct type is
      // ordinal_type.
      const ordinal_type iRow = (static_cast<ordinal_type> (dev.league_rank() * dev.team_size() + dev.team_rank()))
                                * rows_per_thread + loop;
      if (iRow >= m_A.numRows ()) {
        return;
      }
      const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
      const ordinal_type row_length = static_cast<ordinal_type> (row.length);
      value_type sum = 0;

      // Use explicit Cuda below to avoid C++11 for now. This should be a vector reduce loop !
      #ifdef KOKKOS_HAVE_PRAGMA_IVDEP
      #pragma ivdep
      #endif
      #ifdef KOKKOS_HAVE_PRAGMA_UNROLL
      #pragma unroll
      #endif
      #ifdef KOKKOS_HAVE_PRAGMA_LOOPCOUNT
      #pragma loop count (15)
      #endif
#ifdef __CUDA_ARCH__
      for (ordinal_type iEntry = static_cast<ordinal_type> (threadIdx.x);
           iEntry < static_cast<ordinal_type> (row_length);
           iEntry += static_cast<ordinal_type> (blockDim.x)) {
#else
      for (ordinal_type iEntry = 0;
           iEntry < static_cast<ordinal_type> (row_length);
           iEntry ++) {
#endif
        const value_type val = conjugate ?
                ATV::conj (row.value(iEntry)) :
                row.value(iEntry);
        sum += val * m_x(row.colidx(iEntry));
      }

#ifdef __CUDA_ARCH__
      if (blockDim.x > 1)
        sum += Kokkos::shfl_down(sum, 1,blockDim.x);
      if (blockDim.x > 2)
        sum += Kokkos::shfl_down(sum, 2,blockDim.x);
      if (blockDim.x > 4)
        sum += Kokkos::shfl_down(sum, 4,blockDim.x);
      if (blockDim.x > 8)
        sum += Kokkos::shfl_down(sum, 8,blockDim.x);
      if (blockDim.x > 16)
        sum += Kokkos::shfl_down(sum, 16,blockDim.x);

      if (threadIdx.x==0) {
#else
      if (true) {
#endif
        if (doalpha == -1) {
          sum *= value_type(-1);
        } else if (doalpha * doalpha != 1) {
          sum *= alpha;
        }

        if (dobeta == 0) {
          m_y(iRow) = sum ;
        } else if (dobeta == 1) {
          m_y(iRow) += sum ;
        } else if (dobeta == -1) {
          m_y(iRow) = -m_y(iRow) +  sum;
        } else {
          m_y(iRow) = beta * m_y(iRow) + sum;
        }
      }
    }
  }

};

template<typename AType, typename XType, typename YType> 
void matvec_new(AType A, XType x, YType y) {
  typedef typename XType::non_const_value_type Scalar;
  typedef typename AType::execution_space execution_space;
  typedef Kokkos::CrsMatrix<const Scalar,int,execution_space,void,int> matrix_type ;
  typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,execution_space> y_type;
  typedef typename Kokkos::View<const Scalar*,Kokkos::LayoutLeft,execution_space,Kokkos::MemoryRandomAccess > x_type;
  int nnz_per_team = 2048;
  int conc = execution_space::concurrency();
  while((conc * nnz_per_team * 4> A.nnz())&&(nnz_per_team>256)) nnz_per_team/=2;
  int nnz_per_row = A.nnz()/A.numRows();
  //int vector_length = 1;
  //while(vector_length<nnz_per_row/4) vector_length*=2;
  //if(vector_length>32) vector_length = 32;
  //int rows_per_thread = 1;//((nnz_per_team+nnz_per_row-1)/nnz_per_row)/4;
  //int team_size = 256/vector_length;
  int rows_per_team = rows_per_thread * team_size;
  double s_a = 1.0;
  double s_b = 0.0;
  SPMV_Functor<matrix_type,x_type,y_type,1,0,false,int> func (1.0,A,x,0.0,y,rows_per_team);
  //printf("NumRows: %i %i %i %i %i || %i %i\n",y.dimension_0(),A.numRows(),A.nnz(),nnz_per_row,rows_per_team,nnz_per_team,conc);
#ifdef KOKKOS_USE_CUSPARSE
  cusparseDcsrmv (A.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  A.numRows(), A.numCols(),  A.nnz(),
                  &s_a,
                  A.cusparse_descr,
                  A.values.ptr_on_device(),
                  (const int*) A.graph.row_map.ptr_on_device(),
                  A.graph.entries.ptr_on_device(),
                  x.ptr_on_device(),
                  &s_b,
                  y.ptr_on_device());
#else
  int league_size = (y.dimension_0()+rows_per_team-1)/rows_per_team;
  //printf("Vectorlength: %i %i %i\n",vector_length,nnz_per_row,rows_per_team);
  //printf("League_size: %i, Team_size: %i, Vectorlength: %i \n",league_size, team_size, vector_length);
  Kokkos::parallel_for("parallel_for",Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic> >(league_size,team_size,vector_length),func);
#endif
}

/*template<typename Scalar>
int test_crs_matrix_test_multivec(LocalOrdinalType numRows, LocalOrdinalType numCols, LocalOrdinalType nnz, LocalOrdinalType numVecs, LocalOrdinalType test, const char* filename,const bool binaryfile) {
        typedef Kokkos::CrsMatrix<Scalar,LocalOrdinalType,execution_space,void,int> matrix_type ;
        typedef typename Kokkos::MultiVectorDynamic<Scalar,execution_space>::type mv_type;
        typedef typename Kokkos::MultiVectorDynamic<Scalar,execution_space>::random_read_type mv_random_read_type;
        typedef typename mv_type::HostMirror h_mv_type;

        Scalar* val = NULL;
        LocalOrdinalType* row = NULL;
        LocalOrdinalType* col = NULL;

        srand(17312837);
        if(filename==NULL)
          nnz = SparseMatrix_generate<Scalar,LocalOrdinalType>(numRows,numCols,nnz,nnz/numRows*0.2,numRows*0.01,val,row,col);
        else
          if(!binaryfile)
            nnz = SparseMatrix_MatrixMarket_read<Scalar,LocalOrdinalType>(filename,numRows,numCols,nnz,val,row,col);
          else
            nnz = SparseMatrix_ReadBinaryGraph<Scalar,LocalOrdinalType>(filename,numRows,numCols,nnz,val,row,col);

        matrix_type A("CRS::A",numRows,numCols,nnz,val,row,col,false);

        mv_type x("X",numCols,numVecs);
        mv_random_read_type t_x(x);
        mv_type y("Y",numRows,numVecs);
        h_mv_type h_x = Kokkos::create_mirror_view(x);
        h_mv_type h_y = Kokkos::create_mirror_view(y);
        h_mv_type h_y_compare = Kokkos::create_mirror(y);

        typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
        typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);

        for(LocalOrdinalType k=0;k<numVecs;k++){
          //h_a(k) = (Scalar) (1.0*(rand()%40)-20.);
          for(LocalOrdinalType i=0; i<numCols;i++) {
                  h_x(i,k) = (Scalar) (1.0*(rand()%40)-20.);
                  h_y(i,k) = (Scalar) (1.0*(rand()%40)-20.);
          }
        }

        for(LocalOrdinalType i=0;i<numRows;i++) {
                LocalOrdinalType start = h_graph.row_map(i);
                LocalOrdinalType end = h_graph.row_map(i+1);
                for(LocalOrdinalType j=start;j<end;j++) {
                   h_values(j) = h_graph.entries(j) + i;
                }
                for(LocalOrdinalType k = 0; k<numVecs; k++)
                  h_y_compare(i,k) = 0;
                for(LocalOrdinalType j=start;j<end;j++) {
                   Scalar val = h_graph.entries(j) + i;
                   LocalOrdinalType idx = h_graph.entries(j);
                   for(LocalOrdinalType k = 0; k<numVecs; k++)
                           h_y_compare(i,k)+=val*h_x(idx,k);
                }
        }

        Kokkos::deep_copy(x,h_x);
        Kokkos::deep_copy(y,h_y);
        Kokkos::deep_copy(A.graph.entries,h_graph.entries);
        Kokkos::deep_copy(A.values,h_values);

        KokkosSparse::spmv("N",1.0,A,x,0.0,y);
        execution_space::fence();

        Kokkos::deep_copy(h_y,y);
        Scalar error[numVecs];
        Scalar sum[numVecs];
        for(LocalOrdinalType k = 0; k<numVecs; k++) {
                error[k] = 0;
                sum[k] = 0;
        }

        for(LocalOrdinalType i=0;i<numRows;i++) {
          for(LocalOrdinalType k = 0; k<numVecs; k++) {
            error[k]+=(h_y_compare(i,k)-h_y(i,k))*(h_y_compare(i,k)-h_y(i,k));
            sum[k] += h_y_compare(i,k)*h_y_compare(i,k);
          }
        }

        LocalOrdinalType num_errors = 0;
        double total_error = 0;
        double total_sum = 0;
        for(LocalOrdinalType k = 0; k<numVecs; k++) {
                num_errors += (error[k]/(sum[k]==0?1:sum[k]))>1e-5?1:0;
                total_error += error[k];
                total_sum += sum[k];
        }

        LocalOrdinalType loop = 10;
        Kokkos::Impl::Timer timer;
        for(LocalOrdinalType i=0;i<loop;i++)
          KokkosSparse::spmv("N",1.0,A,x,0.0,y);

        execution_space::fence();
        double time = timer.seconds();
        double matrix_size = 1.0*((nnz*(sizeof(Scalar)+sizeof(LocalOrdinalType)) + numRows*sizeof(LocalOrdinalType)))/1024/1024;
        double vector_size = 2.0*numRows*numVecs*sizeof(Scalar)/1024/1024;
        double vector_readwrite = (nnz+numCols)*numVecs*sizeof(Scalar)/1024/1024;

        double problem_size = matrix_size+vector_size;
        printf("%i %i %i %i %6.2lf MB %6.2lf GB/s %6.2lf GFlop/s %6.3lf ms %i\n",
            nnz, numRows,numCols,numVecs,problem_size,(matrix_size+vector_readwrite)/time*loop/1024, 2.0*nnz*numVecs*loop/time/1e9,time/loop*1000, num_errors);
        return (int)total_error;
}*/

template<typename Scalar>
int test_crs_matrix_test_singlevec(int numRows, int numCols, int nnz, int test, const char* filename, const bool binaryfile) {
        typedef Kokkos::CrsMatrix<Scalar,int,execution_space,void,int> matrix_type ;
        typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,execution_space> mv_type;
        typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,execution_space,Kokkos::MemoryRandomAccess > mv_random_read_type;
        typedef typename mv_type::HostMirror h_mv_type;

        Scalar* val = NULL;
        int* row = NULL;
        int* col = NULL;

        srand(17312837);
        if(filename==NULL)
          nnz = SparseMatrix_generate<Scalar,int>(numRows,numCols,nnz,nnz/numRows*0.2,numRows*0.01,val,row,col);
        else
          if(!binaryfile){
            //nnz = SparseMatrix_ExtractBinaryGraph<Scalar, int>(filename, numRows, numCols, nnz, val, row, col);
            nnz = SparseMatrix_MatrixMarket_read<Scalar,int>(filename, numRows, numCols, nnz, val, row, col);
          }else
            nnz = SparseMatrix_ReadBinaryGraph<Scalar,int>(filename,numRows,numCols,nnz,val,row,col);

        matrix_type A("CRS::A",numRows,numCols,nnz,val,row,col,false);

        mv_type x("X",numCols);
        mv_random_read_type t_x(x);
        mv_type y("Y",numRows);
        h_mv_type h_x = Kokkos::create_mirror_view(x);
        h_mv_type h_y = Kokkos::create_mirror_view(y);
        h_mv_type h_y_compare = Kokkos::create_mirror(y);

        typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
        typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);

        for(int i=0; i<numCols;i++) {
                  h_x(i) = (Scalar) (1.0*(rand()%40)-20.);
                  h_y(i) = (Scalar) (1.0*(rand()%40)-20.);
        }

        for(int i=0;i<numRows;i++) {
                int start = h_graph.row_map(i);
                int end = h_graph.row_map(i+1);
                for(int j=start;j<end;j++) {
                   h_values(j) = h_graph.entries(j) + i;
                }
            h_y_compare(i) = 0;
                for(int j=start;j<end;j++) {
                   Scalar val = h_graph.entries(j) + i;
                   int idx = h_graph.entries(j);
                     h_y_compare(i)+=val*h_x(idx);
                }
        }

        Kokkos::deep_copy(x,h_x);
        Kokkos::deep_copy(y,h_y);
        Kokkos::deep_copy(A.graph.entries,h_graph.entries);
        Kokkos::deep_copy(A.values,h_values);

        typename Kokkos::CrsMatrix<Scalar,int,execution_space,void,int>::values_type x1("X1",numCols);
        Kokkos::deep_copy(x1,h_x);
        typename Kokkos::CrsMatrix<Scalar,int,execution_space,void,int>::values_type y1("Y1",numRows);

        int nnz_per_row = A.nnz()/A.numRows();

/*#ifdef MATVEC_NEW
        printf("number of non-zero elements per row %i \n", nnz_per_row);
        while(vector_length <nnz_per_row) vector_length*=2;
        if(vector_length>32) vector_length=32;
        int best_vector_length = vector_length;
        int best_team_size = 32;
        team_size = best_team_size;
        double best_time = 1.e6;
        while(team_size < 1024) {
          while(vector_length <nnz_per_row) vector_length*=2;
          if(vector_length > 1024/team_size) vector_length=1024/team_size;
          while(vector_length > nnz_per_row/8 && vector_length>1) {
            Kokkos::Impl::Timer timer;

            matvec_new(A,x,y);
            execution_space::fence();

            double time = timer.seconds();
            if(time<best_time) { best_time = time; best_vector_length = vector_length; best_team_size = team_size;}
            vector_length/=2;
          }
          team_size*=2;
        }
        vector_length = best_vector_length;
        team_size = best_team_size;

#endif*/

/*#ifdef MATVEC_MINIFE
        matvec_minife(numRows,A.graph.row_map.ptr_on_device(),A.graph.entries.ptr_on_device(),A.values.ptr_on_device(),x1.ptr_on_device(),y1.ptr_on_device());
#endif*/
/*#ifdef MATVEC_NEW
        matvec_new(A,x1,y1);
#endif*/
/*#ifdef MATVEC_TPETRA
        KokkosSparse::spmv("N",1.0,A,x1,0.0,y1);
#endif*/

        Kokkos::deep_copy(h_y,y1);
        Scalar error = 0;
        Scalar sum = 0;
        for(int i=0;i<numRows;i++) {
          error+=(h_y_compare(i)-h_y(i))*(h_y_compare(i)-h_y(i));
          sum += h_y_compare(i)*h_y_compare(i);
          if((h_y_compare(i)-h_y(i))*(h_y_compare(i)-h_y(i))>0)
            if(i==1)
            printf("%i %lf %lf \n",i,h_y_compare(i),h_y(i));
        }

    int num_errors = 0;
    double total_error = 0;
    double total_sum = 0;
    num_errors += (error/(sum==0?1:sum))>1e-5?1:0;
    total_error += error;
    total_sum += sum;
    
//    autoTuning at(&team_size, &vector_length, nnz_per_row);
      printf("nnz_per_row = %i \n", nnz_per_row);

#if (KOKKOS_ENABLE_PROFILING)
    //Kokkos::Profiling::autoTune(&team_size, &vector_length, nnz_per_row, 10);
    //Kokkos::Profiling::autoTune_v2(&team_size, &vector_length, 8);
#endif

    int loop = 100;
    Kokkos::Impl::Timer timer;

        for(int i=0;i<loop;i++) {
#ifdef MATVEC_MINIFE
        matvec_minife(numRows,A.graph.row_map.ptr_on_device(),A.graph.entries.ptr_on_device(),A.values.ptr_on_device(),x1.ptr_on_device(),y1.ptr_on_device());
#endif
#ifdef MATVEC_NEW
//	at.Tuning_start();
        matvec_new(A,x1,y1);
//      at.Tuning_stop();
#endif
#ifdef MATVEC_TPETRA
        KokkosSparse::spmv("N",1.0,A,x1,0.0,y1);
#endif
        }
        execution_space::fence();
        double time = timer.seconds();
        double matrix_size = 1.0*((nnz*(sizeof(Scalar)+sizeof(int)) + numRows*sizeof(int)))/1024/1024;
        double vector_size = 2.0*numRows*sizeof(Scalar)/1024/1024;
        double vector_readwrite = (nnz+numCols)*sizeof(Scalar)/1024/1024;

        double problem_size = matrix_size+vector_size;
    printf("%i %i %i %i %6.2lf MB %6.2lf GB/s %6.2lf GFlop/s %6.3lf ms %i\n",nnz, numRows,numCols,1,problem_size,(matrix_size+vector_readwrite)/time*loop/1024, 2.0*nnz*loop/time/1e9, time/loop*1000, num_errors);
        return (int)total_error;
}


int test_crs_matrix_type(int numrows, int numcols, int nnz, int numVecs, int type, int test, const char* filename, const bool binaryfile) {
  double* val = NULL;
  int* row = NULL;
  int* col = NULL;

  if (!binaryfile){ 
    nnz = SparseMatrix_ExtractBinaryGraph<double, int>(filename, numrows, numcols, nnz, val, row, col);
    return 0;
  }
  //else{
    return test_crs_matrix_test_singlevec<double>(numrows,numcols,nnz,test,filename,binaryfile);
 // }
  //if(numVecs==1)
  //
  //else
  //  return test_crs_matrix_test_multivec<double>(numrows,numcols,nnz,numVecs,test,filename,binaryfile);
}

int main(int argc, char **argv)
{
 long long int size = 110503; // a prime number
 int numVecs = 4;
 int test=-1;
 int type=-1;
 char* filename = NULL;
 bool binaryfile = false;

 rows_per_thread = 1;
 vector_length = 8;
 team_size = 32;
 idx_offset = 0;
 
 for(int i=0;i<argc;i++)
 {
  if((strcmp(argv[i],"-s")==0)) {size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-v")==0)) {numVecs=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--test")==0)) {test=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--type")==0)) {type=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-f")==0)) {filename = argv[++i]; continue;}
  if((strcmp(argv[i],"-fb")==0)) {filename = argv[++i]; binaryfile = true; continue;}
  if((strcmp(argv[i],"-rpt")==0)) {rows_per_thread=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-ts")==0)) {team_size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-vl")==0)) {vector_length=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-offset")==0)) {idx_offset=atoi(argv[++i]); continue;}

 }

 

 Kokkos::initialize(argc,argv);

 int numVecsList[10] = {1, 2, 3, 4, 5, 8, 11, 15, 16, 17};
 int maxNumVecs = numVecs==-1?17:numVecs;
 int numVecIdx = 0;
 if(numVecs == -1) numVecs = numVecsList[numVecIdx++];

 int total_errors = 0;
 while(numVecs<=maxNumVecs) {
   total_errors += test_crs_matrix_type(size,size,size*10,numVecs,type,test,filename,binaryfile);
   if(numVecs<maxNumVecs) numVecs = numVecsList[numVecIdx++];
   else numVecs++;
 }

 if(total_errors == 0)
   printf("Kokkos::MultiVector Test: Passed\n");
 else
   printf("Kokkos::MultiVector Test: Failed\n");


  Kokkos::finalize();
}
