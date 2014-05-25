#include <iostream>
#include <string>
#include <cstdlib>   //ds memory handling
#include <stdexcept> //ds execptions
#include <random>    //ds random numbers
#include <algorithm> //ds lambda functions
#include <chrono>    //ds timing
#include <omp.h>     //ds openmp

//ds types used
typedef double value_type;
typedef std::size_t size_type;

//ds defines
#define ALIGNMENT 64
#define SEED_UNIFORM 42

//ds generates and initializes a fresh matrix
value_type* generateMatrix( value_type* p_matMatrix, const size_type& p_tN, const bool& p_bSetZero = false )
{
    //ds allocate memory for the matrices and escape for the first fail
    if( posix_memalign( reinterpret_cast<void**>( &p_matMatrix ), ALIGNMENT, p_tN*p_tN*sizeof( value_type ) ) )
    {
        //ds escape
        throw std::bad_alloc( );
    }

    //ds check which initialization is desired
    if( p_bSetZero )
    {
        //ds just set all values to zero
        std::generate_n( p_matMatrix, p_tN*p_tN, [ ]( ){ return 0.0; } );
    }
    else
    {
        //ds allocate a random generator with fixed seed for reproduction
        std::mt19937 cGenerator( SEED_UNIFORM );

        //ds allocate a distribution object from 0.0 to 1.0 inclusive
        std::uniform_real_distribution< double > cDistribution ( 0.0, 1.0 );

        //ds initialize values
        std::generate_n( p_matMatrix, p_tN*p_tN, [ &cGenerator, &cDistribution ]( ){ return cDistribution( cGenerator ); } );
    }

    //ds return pointer to memory
    return p_matMatrix;
}

//ds parallel gemm method
void gemm( const size_type& p_tN, const value_type* p_matA, const value_type* p_matB, value_type* p_matC )
{
    //ds parallel
    #pragma omp parallel // shared( p_matC ) firstprivate( p_matA, p_matB, tN )
    {
        //ds loop over each element
        #pragma omp for collapse( 3 )
        for( size_type i = 0; i < p_tN; ++i )
        {
            for( size_type j = 0; j < p_tN; ++j )
            {
                //ds for all k
                for( size_type k = 0; k < p_tN; ++k )
                {
                    //ds value of vice versa
                    p_matC[i*p_tN+j] += p_matA[i*p_tN+k]*p_matB[k*p_tN+j];
                }
            }
        }
    }
}

int main( int argc, char** argv )
{
    //ds first parse input arguments - matrix size [N]
    if( 3 != argc )
    {
        //ds escape
        throw std::runtime_error( "invalid number of arguments: [N] [NUM_THREADS]" );
    }

    //ds set N and threads
    const size_type tN = std::stoul( argv[1] );
    const int iThreads = std::stoul( argv[2] );

    //ds set threads
    omp_set_num_threads( iThreads );

    //ds check if threads could be set
    #pragma omp parallel
    {
        #pragma omp master
        {
            if( iThreads != omp_get_num_threads( ) )
            {
                //ds escape
                throw std::runtime_error( "could not set threads to: " + iThreads );
            }
        }
    }

    std::cout << "      N: " << tN << std::endl;
    std::cout << "Threads: " << iThreads << std::endl;
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

    //ds time gemm
    tmStart = std::chrono::high_resolution_clock::now( );
    gemm( tN, matA, matB, matC );
    tmEnd = std::chrono::high_resolution_clock::now( );

    //ds get time (double resolution is enough)
    const double dDuration = std::chrono::duration< double >( tmEnd - tmStart ).count( );

    //ds inform
    std::cout << "- Computation complete" << std::endl;
    std::cout << "Duration: " << dDuration << " s" << std::endl;

    return 0;
}
