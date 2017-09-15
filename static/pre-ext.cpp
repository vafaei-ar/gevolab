#include <stdlib.h>
#include "LATfield2.hpp"

using namespace LATfield2;

int main(int argc, char **argv)
{
int i,n,m;
string filename;
string outname = "pre-out";
long int npart_lim=0;

m = 2;
n = 2;

	for (i=1 ; i < argc ; i++ ){
		if ( argv[i][0] != '-' )
			continue;
		switch(argv[i][1]) {
			case 'i':
				npart_lim = atoi(argv[++i]); //size of the dim 1 of the processor grid
				break;
			case 's':
				filename = argv[++i]; //snap file name
				break;
		}
	}

parallel.initialize(n,m);

Lattice lat_part(3,16,0);

part_simple_info particles_global_info;
part_simple_dataType particles_dataType;

part_simple part;

struct fileDsc fd[1];
get_fileDsc_global(filename + ".h5",fd[0]);
get_partInfo(filename+".h5",particles_global_info,particles_dataType);
   
// Here is the iteration over all particles and you can adjust every given condition

// Box length
fd[0].boxSize[0] = 1;
fd[0].boxSize[1] = 1;
fd[0].boxSize[2] = 1;

//Determining the number of data

parallel.barrier();

Particles<part_simple,part_simple_info,part_simple_dataType> parts;
parts.initialize(particles_global_info,particles_dataType,&lat_part,fd[0].boxSize);

for(i=0;i<npart_lim;i++)
{
			part.ID = 0;
			part.pos[0]= 0.5;
			part.pos[1]= 0.5;
			part.pos[2]= 0.5;
			part.vel[0] = 0;
			part.vel[1] = 0;
			part.vel[2] = 0;
			parts.addParticle_global(part);
}

parts.saveHDF5(outname,1);
//cout<<particles_global_info.mass<<endl;
//cout<<particles_global_info.relativistic<<endl;
//cout<<fd[0].boxSize[0]<<endl;
}
