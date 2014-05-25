#include <iostream>
#include <string>
#include <cstdlib>   //ds memory handling
#include <stdexcept> //ds execptions
#include <random>    //ds random numbers
#include <algorithm> //ds lambda functions
#include <chrono>    //ds timing

//ds blas function
extern "C" void dgemm_( char& TRANSA, char& TRANSB, int& M, int& N, int& K, double& ALPHA, double* A, int& LDA, double* B, int& LDB, double& BETA, double* C, int& LDC );

//ds types used
typedef double value_type;
typedef std::size_t size_type;

//ds defines
#define ALIGNMENT 64
#define SEED_UNIFORM 42

//ds generates and initializes a fresh matrix
value_type* generateMatrix( value_type* p_matMatrix, const size_type tN, const bool p_bSetZero = false )
{
    //ds allocate memory for the matrices and escape for the first fail
    if( posix_memalign( reinterpret_cast<void**>( &p_matMatrix ), ALIGNMENT, tN*tN*sizeof( value_type ) ) )
    {
        //ds escape
        throw std::bad_alloc( );
    }

    //ds check which initialization is desired
    if( p_bSetZero )
    {
        //ds just set all values to zero
        std::generate_n( p_matMatrix, tN*tN, [ ]( ){ return 0.0; } );
    }
    else
    {
        //ds allocate a random generator with fixed seed for reproduction
        std::mt19937 cGenerator( SEED_UNIFORM );

        //ds allocate a distribution object from 0.0 to 1.0 inclusive
        std::uniform_real_distribution< double > cDistribution ( 0.0, 1.0 );

        //ds initialize values
        std::generate_n( p_matMatrix, tN*tN, [ &cGenerator, &cDistribution ]( ){ return cDistribution( cGenerator ); } );
    }

    //ds return pointer to memory
    return p_matMatrix;
}

int main( int argc, char** argv )
{
    //ds first parse input arguments - matrix size [N]
    if( 2 != argc )
    {
        //ds escape
        throw std::runtime_error( "invalid number of arguments: [N]" );
    }

    //ds set N
    const size_type tN = std::stoul( argv[1] );

    std::cout << "N: " << tN << std::endl;
    std::cout << "- Starting initialization .." << std::endl;

    //ds define 3 matrices A, B, C to perform C = AxB
    value_type* matA = 0;
    value_type* matB = 0;
    value_type* matC = 0;

    //ds allocate memory for the matrices and initialize them
    matA = generateMatrix( matA, tN );
    matB = generateMatrix( matB, tN );
    matC = generateMatrix( matC, tN, true );

    std::cout << "- Initialization complete" << std::endl;
    std::cout << "- Starting computation .." << std::endl;

    //ds timing
    std::chrono::time_point< std::chrono::high_resolution_clock > tmStart, tmEnd;

    //ds blas configuration (2 times transposed because of col major in blas)
    double dAlpha    = 1.0;
    double dBeta     = 0.0;
    char cTransposeA = 'T';
    char cTransposeB = 'T';
    int iN           = static_cast< int >( tN );

    //ds time gemm
    tmStart = std::chrono::high_resolution_clock::now( );
    dgemm_( cTransposeA, cTransposeB, iN, iN, iN, dAlpha, matA, iN, matB, iN, dBeta, matC, iN );
    tmEnd = std::chrono::high_resolution_clock::now( );

    //ds IMPORTANT: matC has now FORTRAN INDICES (i,j) -> (j,i)

    //ds get time (double resolution is enough)
    const double dDuration = std::chrono::duration< double >( tmEnd - tmStart ).count( );

    //ds inform
    std::cout << "- Computation complete" << std::endl;
    std::cout << "Duration: " << dDuration << " s" << std::endl;

    return 0;
}
