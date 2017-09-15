import numpy as np
import subprocess
import h5py
import os

##########################################################
#                       FUNCTIONS
##########################################################

# parser function for setting file
def parser(path):
    lst = []
    for line in open(path, 'r'):
        line = line.split('#', 1)[0]
        line = line.rstrip()
        if line!='':
            lst.append(line.split('='))
    res = {cs[0].strip():s2n(cs[1].split(',')) for cs in lst}
    root = path.split('/')[:-1]
    root = "/".join(root)
    oup = root+'/'+res['output path'][0]
    res['root'] = root
    res['outdir'] = oup
#     factsheet(res)
    return res

# list string to number convertor
def s2n(lst):
    lstp = []
    for i in lst:
        try:
            lstp.append(float(i))
        except :
            lstp.append(i.strip())
    return lstp

# making fact sheet of setting
def factsheet(setting):
    print ('fact sheet:')
    for keys,values in setting.items():
        print (keys+': ')
        
# general job script generator
def job_script_make(exe,n_core=16,time='5:00:00',opts=''):
    l_c = int(np.sqrt(n_core))
    if l_c**2==n_core:
        scr_txt='\
#!/bin/bash\n\
\n\
#SBATCH -J name\n\
#SBATCH --get-user-env\n\
#SBATCH --ntasks='+str(n_core)+'\n\
#SBATCH --cpus-per-task=1\n\
#SBATCH -p dpt\n\
#SBATCH --output=slurm-test-%J.out\n\
#SBATCH -t '+time+'\n\
#SBATCH --mail-type=FAIL\n\
\n\
##cd ##ROOTDIR##\n\
#source /etc/profile.modules\n\
module load foss/2016a\n\
module load GSL/1.16\n\
module unload openmpi/intel/184\n\
module load openmpi/gcc/1102\n\
echo Running on host `hostname`\n\
echo Time is `date`\n\
echo Directory is `pwd`\n\
echo Slurm job ID is $SLURM_JOBID\n\
\n\
#export OMP_NUM_THREADS=4\n\
\n\
##COMMAND##\n\
time srun -n '+str(n_core)+' '+exe+' -n '+str(l_c)+' -m '+str(l_c)+' '+opts+'\n\
\n\
#wait for processes to finish\n\
wait\n\
echo End time is `date`\n\
'
        file = open('gevolab_run.sh','w') 
        file.write(scr_txt) 
        file.close()
        subprocess.call(['chmod','700', 'gevolab_run.sh'])
        
# general function for submitting a script
def qsub(script):
    subprocess.call(['qsub',script])

# script generator and submitter
def scr_sub(exe,n_core=16,time='5:00:00',opts=''):
    job_script_make(exe,n_core=n_core,time=time,opts=opts)
    qsub('gevolab_run.sh')

# halo extractor and rescaler
def extractor(file_name,id_add,outname,function=None):

    exe = __file__.split('/')[:-2]
    exe = '/'.join(exe)+'/exe/pre-ext'
    if file_name[-3:]=='.h5':
        file_name = file_name[:-3]
    if outname[-3:]=='.h5':
        outname = outname[:-3]

    f_snap0 = h5py.File(file_name+'.h5', 'r')
    snap0 = f_snap0['data']

    IDs = np.loadtxt(id_add)
    mask = np.in1d(snap0['ID'],IDs)

    subprocess.call(['mpirun', '-np', '4', exe, '-s', file_name, '-i', str(IDs.shape[0])])

    file_name = 'pre-out.h5'
    f1 = h5py.File(file_name, 'r+')

    if function==None:
        f1['data']['positionX'] =snap0[mask]['positionX']
        f1['data']['positionY'] =snap0[mask]['positionY']
        f1['data']['positionZ'] =snap0[mask]['positionZ']
    else:
        f1['data']['positionX'] =function(snap0[mask]['positionX'])
        f1['data']['positionY'] =function(snap0[mask]['positionY'])
        f1['data']['positionZ'] =function(snap0[mask]['positionZ'])

        n_out = np.any((f1['data']['positionX']<=0,\
        f1['data']['positionX']>=1,\
        f1['data']['positionY']<=0,\
        f1['data']['positionY']>=1,\
        f1['data']['positionZ']<=0,\
        f1['data']['positionZ']>=1), axis=0).sum()
        if n_out!=0 :
            print 'Warning!'
            print '%d particles went out by transformation!' %n_out

    f1['data']['velocityX'] =snap0[mask]['velocityX']
    f1['data']['velocityY'] =snap0[mask]['velocityY']
    f1['data']['velocityZ'] =snap0[mask]['velocityZ']
    f1['data']['ID'] =snap0[mask]['ID']        

    poc_0 = np.logical_and(f1['data']['positionY']<0.5,f1['data']['positionZ']<0.5)
    poc_1 = np.logical_and(f1['data']['positionY']<0.5,f1['data']['positionZ']>=0.5)
    poc_2 = np.logical_and(f1['data']['positionY']>=0.5,f1['data']['positionZ']<0.5)
    poc_3 = np.logical_and(f1['data']['positionY']>=0.5,f1['data']['positionZ']>=0.5)

    n1 = poc_0.sum()
    n2 = poc_1.sum()
    n3 = poc_2.sum()
    n4 = poc_3.sum()

    intype = type(f1['numParts'][0])
    n1 = np.array([n1], dtype=intype)[0]
    n2 = np.array([n2], dtype=intype)[0]
    n3 = np.array([n3], dtype=intype)[0]
    n4 = np.array([n4], dtype=intype)[0]
    n_tot = n1+n2+n3+n4

    proc_list = np.zeros(n_tot)
    proc_list[poc_0] = 0
    proc_list[poc_1] = 1
    proc_list[poc_2] = 2
    proc_list[poc_3] = 3

    sorted_proc_list = proc_list.argsort()
    f1['data']['ID'] = f1['data']['ID'][sorted_proc_list]
    f1['data']['positionX'] = f1['data']['positionX'][sorted_proc_list]
    f1['data']['positionY'] = f1['data']['positionY'][sorted_proc_list]
    f1['data']['positionZ'] = f1['data']['positionZ'][sorted_proc_list]
    f1['data']['velocityX'] = f1['data']['velocityX'][sorted_proc_list]
    f1['data']['velocityY'] = f1['data']['velocityY'][sorted_proc_list]
    f1['data']['velocityZ'] = f1['data']['velocityZ'][sorted_proc_list]

    f1['numParts'][0] = n1
    f1['numParts'][1] = n2
    f1['numParts'][2] = n3
    f1['numParts'][3] = n4

    f1.close()

    subprocess.call(['mv', 'pre-out.h5',outname+'.h5'])

    print('Done!')

# plot a snapshot and save
def snap_save(snap,n_sample=1000):
    import matplotlib.pylab as plt
    import mpl_toolkits.mplot3d.axes3d as p3

    if snap[-3:]!='.h5':
        snap = snap+'.h5'

    f = h5py.File(snap, "r")
    n_parts = np.sum(np.array(f['numParts']))
    if n_sample<n_parts:
        pl = np.arange(n_parts)
        np.random.shuffle(pl)
        pl = pl[:n_sample]
    else:
        pl = np.arange(n_parts)
    sn = np.array(list(f['data']))


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # NOTE: Can't pass empty arrays into 3d version of plot()
    ax.plot(sn['positionX'][pl], sn['positionY'][pl], sn['positionZ'][pl], 'ko')

    # Setting the axes properties
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')

    ax.set_title('')

    plt.savefig(snap[:-3]+'.jpg')
#     plt.show()

# plot a snapshot and show
def snap_show(snap,n_sample=1000):
    import matplotlib.pylab as plt
    import mpl_toolkits.mplot3d.axes3d as p3

    if snap[-3:]!='.h5':
        snap = snap+'.h5'

    f = h5py.File(snap, "r")
    n_parts = np.sum(np.array(f['numParts']))
    if n_sample<n_parts:
        pl = np.arange(n_parts)
        np.random.shuffle(pl)
        pl = pl[:n_sample]
    else:
        pl = np.arange(n_parts)
    sn = np.array(list(f['data']))

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # NOTE: Can't pass empty arrays into 3d version of plot()
    ax.plot(sn['positionX'][pl], sn['positionY'][pl], sn['positionZ'][pl], 'ko')

    # Setting the axes properties
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')
    ax.set_title('')
    
    plt.show()

def depth(L):
    return isinstance(L, list) and max(map(depth, L))+1

def snap_movie(lst,outname,n_sample=1000):
    import matplotlib.pylab as plt
    import mpl_toolkits.mplot3d.axes3d as p3

    clrs = ['k','r','b','orange','y'] 

    n_dpth = depth(lst)
    if n_dpth==1:
        n_snap = len(lst)
        n_halos = 1
    elif n_dpth==2:
        n_snap = len(lst[0])
        n_halos = len(lst)
    else:
        print('snap structure is wrong!')
        return

    fig = plt.figure()
    ax = p3.Axes3D(fig)

# Setting the axes properties
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 1.0])
    ax.set_zlabel('Z')
    ax.set_title('')

    outf = outname+'.avi'
    rate = 2

    cmdstring = ('ffmpeg',
        '-r', '%d' % rate,
        '-s', '600x600',
        '-f','image2pipe',
        '-vcodec', 'png',
        '-i', 'pipe:', outf
        )
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    frame = [None for i in range(n_halos)]
    pl = [None for i in range(n_halos)]

    for snap_i in range(n_snap):
        for halo_i in range(n_halos):

            if n_dpth==1:
                snap = lst[snap_i]
            elif n_dpth==2:
                snap = lst[halo_i][snap_i]

            f = h5py.File(snap, "r")
            srt = np.argsort(f['data']['ID'])

            if snap_i==0:
                n_parts = np.sum(np.array(f['numParts']))
                if n_sample<n_parts:
                    pl[halo_i] = np.arange(n_parts)
                    np.random.shuffle(pl[halo_i])
                    pl[halo_i] = pl[halo_i][:n_sample]
                else:
                    pl[halo_i] = np.arange(n_parts)
            sn = np.array(list(f['data']))

            frame[halo_i], = ax.plot(sn['positionX'][srt][pl[halo_i]], \
                                     sn['positionY'][srt][pl[halo_i]], \
                                     sn['positionZ'][srt][pl[halo_i]], \
                                     'o', color = clrs[halo_i])

        plt.savefig(p.stdin, format='png',dpi=300)

        for halo_i in range(n_halos):
            frame[halo_i].remove()

    p.stdin.close()

# halo extractor and rescaler
def h5_generator(pos,vel,outname,mass=1,rel=0):

    if len(pos.shape)!=2 or pos.shape[1]!=3:
        print ('Wrong format for positions!')
        return
    if len(vel.shape)!=2 or vel.shape[1]!=3:
        print ('Wrong format for velocities!')
        return

    n_pos = pos.shape[0]
    n_vel = vel.shape[0]
    if n_pos!=n_vel:
        print ('Number of positions should be equal to the number of velocities!')
        return

    exe = __file__.split('/')[:-2]
    exe = '/'.join(exe)+'/exe/pre-h5_gen'
    if outname[-3:]=='.h5':
        outname = outname[:-3]

    subprocess.call(['mpirun', '-np', '4', exe, '-i', str(n_pos)])

    file_name = 'pre-out.h5'
    f1 = h5py.File(file_name, 'r+')

    f1['part_info']['mass'] = np.array([mass])
    f1['part_info']['relativistic'] = np.array([rel])
#    f1['part_info']['type_name'][0] = 'part_simple'

    f1['data']['positionX'] = pos[:,0]
    f1['data']['positionY'] = pos[:,1]
    f1['data']['positionZ'] = pos[:,2]

    f1['data']['velocityX'] = vel[:,0]
    f1['data']['velocityY'] = vel[:,1]
    f1['data']['velocityZ'] = vel[:,2]
    f1['data']['ID'] = np.arange(n_pos)     

    poc_0 = np.logical_and(f1['data']['positionY']<0.5,f1['data']['positionZ']<0.5)
    poc_1 = np.logical_and(f1['data']['positionY']<0.5,f1['data']['positionZ']>=0.5)
    poc_2 = np.logical_and(f1['data']['positionY']>=0.5,f1['data']['positionZ']<0.5)
    poc_3 = np.logical_and(f1['data']['positionY']>=0.5,f1['data']['positionZ']>=0.5)

#    onb = np.array([])
#    for i in range(3):
#        pos_l = np.argwhere(pos[:,i]==0.5).flatten()
#        onb = np.append(onb,pos_l)

#    pos = np.delete(pos, onb, 0) 

    n1 = poc_0.sum()
    n2 = poc_1.sum()
    n3 = poc_2.sum()
    n4 = poc_3.sum()

    intype = type(f1['numParts'][0])
    n1 = np.array([n1], dtype=intype)[0]
    n2 = np.array([n2], dtype=intype)[0]
    n3 = np.array([n3], dtype=intype)[0]
    n4 = np.array([n4], dtype=intype)[0]
    n_tot = n1+n2+n3+n4

    proc_list = np.zeros(n_tot)
    proc_list[poc_0] = 0
    proc_list[poc_1] = 1
    proc_list[poc_2] = 2
    proc_list[poc_3] = 3

    sorted_proc_list = proc_list.argsort()
    f1['data']['ID'] = f1['data']['ID'][sorted_proc_list]
    f1['data']['positionX'] = f1['data']['positionX'][sorted_proc_list]
    f1['data']['positionY'] = f1['data']['positionY'][sorted_proc_list]
    f1['data']['positionZ'] = f1['data']['positionZ'][sorted_proc_list]
    f1['data']['velocityX'] = f1['data']['velocityX'][sorted_proc_list]
    f1['data']['velocityY'] = f1['data']['velocityY'][sorted_proc_list]
    f1['data']['velocityZ'] = f1['data']['velocityZ'][sorted_proc_list]

    f1['numParts'][0] = n1
    f1['numParts'][1] = n2
    f1['numParts'][2] = n3
    f1['numParts'][3] = n4

    f1.close()

    subprocess.call(['mv', 'pre-out.h5',outname+'.h5'])

##########################################################
#                         CLASSES
##########################################################

class gev_analyze:
    def __init__(self,setting_path):
        self.__setting_path = setting_path
        self.root = '/'.join(setting_path.split('/')[:-1])+'/'
        self.__setting = parser(setting_path)
        self.__setting_name = self.__setting_path.split('/')[-1]
        self.snap_number = len(self.__setting['snapshot redshifts'])
        self.pk_number = len(self.__setting['Pk outputs'])
        self.hostpath = None

        pre = __file__.split('/')[:-2]
        self.rs_add = '/'.join(pre)+'/static/rockstar/rockstar'
#         self.remote_path = None
        if not os.path.exists(self.root+'/gevolution'):
            print 'Gevolution is not found!'
            print 'Default version will be used.'
            print 
            pre = __file__.split('/')[:-2]
            self.gev_path = '/'.join(pre)+'/static/gevolution/'
            self.gev_def = True
        else:
            self.gev_path = self.root
            self.gev_def = False

    def ins_name(self):
        return [k for k,v in globals().items() if v is self][0]

    def setting_getter(self):
        return self.__setting
    
    def load(self,typ,name,nfile):
        """This method is for loading requested outputs of 
        desired setting file which is specified in instance.
        inputs:
        1- typ is type of requested file (snap or pk)
        2- name is requested filed (for example phi, chi, etc)
        3- nfile is the snapshot number of requested filed."""

        if not (typ=='snap' or typ=='pk'):
            print ("The first argument should be 'snap' or 'pk'.")
            return

        if ((typ=='snap') & (name in self.__setting['snapshot outputs'])):
            nsnap = nfile
            if (nsnap<self.snap_number):
                fil = self.__setting['outdir']+self.__setting['snapshot file base'][0]+\
                str(format(nsnap,'03'))+'_'+name+'.h5'
                if name=='pcls':
                    fil = self.__setting['outdir']+self.__setting['snapshot file base'][0]+\
                    str(format(nsnap,'03'))+'_'+'cdm'+'.h5'
                f = h5py.File(fil, "r")
                s = np.array(list(f['data']))
            else:
                print ('Invalid request!')
                print ('Valid cases are:')
                print ('Number of requested snapshot should be less than '+str(self.snap_number))
                print 
                return
            
        if ((typ=='pk') & (name in self.__setting['Pk outputs'])):
            npk = nfile
            if (npk<self.pk_number):
                fil = self.__setting['outdir']+self.__setting['Pk file base'][0]+\
                str(format(npk,'03'))+'_'+name+'.dat'
                if name=='pcls':
                    fil = self.__setting['outdir']+self.__setting['Pk file base'][0]+\
                    str(format(npk,'03'))+'_'+'cdm'+'.dat'
                s = np.loadtxt(fil)
            else:
                print ('Invalid request!')
                print ('Valid cases are:')
                print ('Number of requested power spectrum should be less than '+str(self.pk_number))
                print 
                return
            
        if not ((name in self.__setting['snapshot outputs']) | (name in self.__setting['Pk outputs'])):
            print ('Invalid request!')
            print ('Valid cases for snapshots are:')
            print (self.__setting['snapshot outputs'])
            print ('Valid cases for power spectrum are:')
            print (self.__setting['Pk outputs'])
            print 
            return       
        return s
    
    def snap_list(self):

        return [self.__setting['snapshot file base'][0]+\
        str(format(nsnap,'03'))+'_'+'cdm'+'.h5'\
        for nsnap in range(self.snap_number)]

    def push(self,filename):
        try:
            instnc = self.ins_name()
        except:
            instnc = '$local_variable_of_class'
        
        if self.hostpath==None:
            print 'You didn\'t set hostname, please set your host name in a variable named <'+instnc+'.hostname>.'
            return
        subprocess.call(['scp', filename, self.hostpath])

    def outputs_push(self,typ,name,nfile):
        try:
            instnc = self.ins_name()
        except:
            instnc = '$local_variable_of_class'
        nsnap = nfile
        
        if self.hostpath==None:
            print 'You didn\'t set hostname, please set your host name in a variable named <'+instnc+'.hostname>.'
            return

        if ((name in self.__setting['snapshot outputs']) & (nsnap<self.snap_number)):
            
            if (typ=='snap'):
                fil = self.__setting['outdir']+self.__setting['snapshot file base'][0]+\
                str(format(nsnap,'03'))+'_'+name+'.h5'
                if name=='pcls':
                    fil = self.__setting['outdir']+self.__setting['snapshot file base'][0]+\
                    str(format(nsnap,'03'))+'_'+'cdm'+'.h5'
#                 print ['scp', fil, self.hostpath]
                subprocess.call(['scp', fil, self.hostpath])

            if (typ=='pk'):
                npk = nfile
                if (npk<self.pk_number):
                    fil = self.__setting['outdir']+self.__setting['Pk file base'][0]+\
                    str(format(npk,'03'))+'_'+name+'.dat'
                    if name=='pcls':
                        fil = self.__setting['outdir']+self.__setting['Pk file base'][0]+\
                        str(format(npk,'03'))+'_'+'cdm'+'.dat'
#                 print ['scp', fil, self.hostpath]
                subprocess.call(['scp', fil, self.hostpath])

        else:
            print ('Invalid request!')
            print ('Valid cases are:')
            
            print ('snapshots:')
            print (self.__setting['snapshot outputs'])
            print ('Number of requested snapshot should be less than '+str(self.snap_number))

            print ('Pks:')
            print (self.__setting['Pk outputs'])
            print ('Number of requested powerspectrums should be less than '+str(self.pk_number))

            return
        
    def rockstar(self,nsnap):  
        
        if 'Gadget2' not in self.__setting['snapshot outputs']:
            print ('Given setting file doesn\'t include Gadjet2 output.')
            print 
            return   
        
        if (nsnap>=self.snap_number):
                print ('Invalid request!')
                print ('Number of requested snapshot should be less than '+str(self.snap_number))
                print 
                return    
            
        name = str(format(nsnap,'03'))+'_cdm'
        
        self.rs_cfg_make()
        
        print 'Rockstar Halo Finder:\n\
This fucntion is adjusted for single-cpu, single snapshot halo finding.\n\
Note that non-periodic boundary conditions are assumed.\
See Rockstar README for more details.'
        print 'Please wait ...'
        print 'Caution! If you had any problem with Cosmologies with Curvature, you can crack main.cpp  in gevolution!'
        print 

        fil = self.__setting['outdir']+self.__setting['snapshot file base'][0]+name
        subprocess.call([self.rs_add, '-c','gev.cfg', fil])
        
        labels = ['id', 'num_p', 'mvir', 'mbound_vir', 'rvir', 'vmax', 'rvmax', 'vrms', 'x', 'y', \
 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz', 'E', 'Spin', 'PosUncertainty', \
 'VelUncertainty', 'bulk_vx', 'bulk_vy', 'bulk_vz', 'BulkVelUnc', 'n_core', \
 'm200b', 'm200c', 'm500c', 'm2500c', 'Xoff', 'Voff', 'spin_bullock', 'b_to_a', \
 'c_to_a', 'A[x]', 'A[y]', 'A[z]', 'b_to_a(500c)', 'c_to_a(500c)', 'A[x](500c)', \
 'A[y](500c)', 'A[z](500c)', 'Rs', 'Rs_Klypin', 'T/|U|', 'M_pe_Behroozi', \
 'M_pe_Diemer', 'idx', 'i_so', 'i_ph', 'num_cp', 'mmetric']

        formats = 53*[np.float32]
        
#         if subprocess.call(['ls', 'halos_0.0.ascii'])==0:
        if os.path.exists('halos_0.0.ascii'):
            print 'Returned quantities are: ',labels
            
            return np.loadtxt('halos_0.0.ascii', comments='#', dtype={'names': labels,\
                 'formats': formats})
        else:
            print 'Something is wrong!'
            print 
            return
    def rockstar_bin(self,ky):
#         if subprocess.call(['ls', 'halos_0.0.bin'])==0:
        if os.path.exists('halos_0.0.bin'):    
            print 'For new snap number run rockstar again.'
            
            import readgadget as ggr
            try:
                return ggr.readrockstar('halos_0',ky)   
            except:
                print 'Error!'
                print 'Requested key may be wrong!'
                print 'See https://bitbucket.org/rthompson/pygadgetreader'
                print 
                return

        else:
            print 'Run Rockstar method first!'
            print 
            return
        
    def rs_cfg_make(self):
        scr_txt = '\
# This file is created by Gevolab\n\
\n\
FILE_FORMAT = "GADGET2"\n\
\n\
# particle data file\n\
SCALE_NOW = 1\n\
h0 = '+str(self.__setting['h'][0])+'\n\
Ol = '+str(1-(self.__setting['omega_cdm'][0]+self.__setting['omega_b'][0]))+'\n\
Om = '+str((self.__setting['omega_cdm'][0]+self.__setting['omega_b'][0]))+'\n\
\n\
GADGET_LENGTH_CONVERSION = 0.001\n\
#GADGET_MASS_CONVERSION = 0.84849e10\n\
GADGET_MASS_CONVERSION = 1.0e10\n\
#GADGET_VELOCITY_CONVERSION 3.335640952e-6\n\
#actual mass 9.45799e8\n\
\n\
FORCE_RES = '+str((1.0*self.__setting['boxsize'][0])/self.__setting['Ngrid'][0])+\
' #Force resolution of simulation, in Mpc/h\n'
        file = open('gev.cfg','w') 
        file.write(scr_txt) 
        file.close() 
        
    def job_submit(self,n_core=16,time='5:00:00',cls=False):

        rqrs = ['bcc_crystal.dat',\
        'class_tk.dat',\
        'fcc_crystal.dat',\
        'gevolution',\
        'sc0_crystal.dat',\
        'sc1_crystal.dat',]

        if self.gev_def:
            add0 = self.gev_path
            for fl in rqrs:
                subprocess.call(['cp',add0+fl,self.root])

        if not os.path.isdir(self.__setting['outdir']):
            subprocess.call(['mkdir', '-p', self.__setting['outdir']])
            
        add0 = os.getcwd()
        os.chdir(self.root)
        scr_sub('gevolution',n_core=n_core,time=time,opts='-s '+self.__setting_name)
        os.chdir(add0)

        if (self.gev_def and cls):
            import time as tm
            tm.sleep(5) # delays
            for fl in rqrs:
                subprocess.call(['rm',self.root+fl])

#         subprocess.call(['scp', filename, ':'.join([self.hostname,self.root+self.__setting['outdir']])])

    def local_run(self,n1=2,n2=2):

        rqrs = ['bcc_crystal.dat',\
        'class_tk.dat',\
        'fcc_crystal.dat',\
        'gevolution',\
        'sc0_crystal.dat',\
        'sc1_crystal.dat',]

        if self.gev_def:
            add0 = self.gev_path
            for fl in rqrs:
                subprocess.call(['cp',add0+fl,self.root])

        n1,n2 = int(n1), int(n2)
        if not os.path.isdir(self.__setting['outdir']):
            subprocess.call(['mkdir', '-p', self.__setting['outdir']])
            
        add0 = os.getcwd()
        os.chdir(self.root)
            
        subprocess.call(['mpirun','-np', str(n1*n2), './gevolution','-n', str(n1), '-m', str(n2), '-s', self.__setting_name])
        os.chdir(add0)

        if self.gev_def:
            for fl in rqrs:
                subprocess.call(['rm',self.root+fl])

    def duplicate(self, filename, **kwargs):
        if filename==self.__setting_path:
            print ("Error! Duplication can't overwrite file!")
            return

        if kwargs is not None:
            for key, value in kwargs.iteritems():
                self.__setting[key]=[value]


        sep = ','
        keys = self.__setting.keys()
        keys.remove('root')
        keys.remove('outdir')
        with open(filename, "w") as f:
            f.write("# This file is duplicated by Gevolab.\n\n")
            for i in keys:            
                f.write(i + " = " + sep.join([str(x) for x in self.__setting[i]]) + "\n")
                
        self.__setting = parser(self.__setting_path)

    def snap_movie(self,n_sample=1000):
        import matplotlib.pylab as plt
        import mpl_toolkits.mplot3d.axes3d as p3
#        from matplotlib import gridspec
#        
        fig = plt.figure()
#        fig.set_size_inches(6, 6)

#        gs = gridspec.GridSpec(1, 1)
#        ax = fig.add_subplot(gs[0])

        ax = p3.Axes3D(fig)
        ax.view_init(0,0)

    # Setting the axes properties
        ax.set_xlim3d([0.0, 1.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([0.0, 1.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 1.0])
        ax.set_zlabel('Z')
        ax.set_title('')

        outf = self.__setting_name+'.avi'
        rate = 2

        cmdstring = ('ffmpeg',
            '-r', '%d' % rate,
            '-s', '600x600',
            '-f','image2pipe',
            '-vcodec', 'png',
            '-i', 'pipe:', outf
            )
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        for snap in range(self.snap_number):
            sn = self.load('snap','pcls',snap)
            srt = np.argsort(sn['ID'])
            if snap==0:
                n_parts = sn.shape[0]
                if n_sample<n_parts:
                    pl = np.arange(n_parts)
                    np.random.shuffle(pl)
                    pl = pl[:n_sample]
                else:
                    pl = np.arange(n_parts)

    # NOTE: Can't pass empty arrays into 3d version of plot()
            p_0, = ax.plot(sn['positionX'][srt][pl], \
                           sn['positionY'][srt][pl], \
                           sn['positionZ'][srt][pl], 'ko')

            plt.savefig(p.stdin, format='png',dpi=300)
            p_0.remove()

        p.stdin.close()





