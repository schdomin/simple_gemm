#include <iostream>
#include <string>
#include <cstdlib>   //ds memory handling
#include <stdexcept> //ds execptions
#include <random>    //ds random numbers
#include <algorithm> //ds lambda functions
#include <chrono>    //ds timing
#include <thread>    //ds c++11 threads

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

//ds row major
void gemm_parallel( const size_type& p_tN, const size_type& p_tRowStart, const size_type& p_tRowEnd, const value_type* p_matA, const value_type* p_matB, value_type* p_matC )
{
    //ds loop over our row part
    for( size_type i = p_tRowStart; i < p_tRowEnd; ++i )
    {
        //ds do all columns
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

//ds main gemm method
void gemm( const size_type& p_tN, const size_type& p_tThreads, const value_type* p_matA, const value_type* p_matB, value_type* p_matC )
{
    //ds compute work chunks (simple row division)
    const size_type tRowsPerThread = p_tN/p_tThreads;

    //ds escape for invalid values
    if( p_tN != tRowsPerThread*p_tThreads )
    {
        //ds escape
        throw std::runtime_error( "invalid parameters" );
    }

    //ds allocate our thread buddies
    std::vector< std::thread > vecThreads( p_tThreads );

    //ds divide work over threads
    for( size_type i = 0; i < p_tThreads; ++i )
    {
        //ds compute start and end position
        const size_type tRowStart = i*tRowsPerThread;
        const size_type tRowEnd   = tRowStart+tRowsPerThread;

        //ds info
        std::cout << "- Thread[" << i << "] assigned rows: " << tRowStart << " to " << tRowEnd << std::endl;

        //ds hand the parallel work to thread
        vecThreads[i] = std::thread( gemm_parallel, p_tN, tRowStart, tRowEnd, p_matA, p_matB, p_matC );
    }

    //ds wait till all threads finished
    for( std::thread& cThread: vecThreads )
    {
        //ds join
        cThread.join( );
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
    const size_type tN       = std::stoul( argv[1] );
    const size_type tThreads = std::stoul( argv[2] );

    //ds get available threads
    const size_type tThreadsAvailable = std::thread::hardware_concurrency( );

    std::cout << "      N: " << tN << std::endl;
    std::cout << "Threads: " << tThreads << "/" << tThreadsAvailable << std::endl;
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

    //ds time gemm (it will take care of the threading)
    tmStart = std::chrono::high_resolution_clock::now( );
    gemm( tN, tThreads, matA, matB, matC );
    tmEnd = std::chrono::high_resolution_clock::now( );

    //ds get time (double resolution is enough)
    const double dDuration = std::chrono::duration< double >( tmEnd - tmStart ).count( );

    //ds inform
    std::cout << "- Computation complete" << std::endl;
    std::cout << "Duration: " << dDuration << " s" << std::endl;

    return 0;
}
