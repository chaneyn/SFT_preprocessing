import warnings
warnings.filterwarnings('ignore')
#import gdal
import os
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.stdout.flush()

import geospatialtools.pedotransfer as pedotransfer
import geospatialtools.gdal_tools as gdal_tools
import geospatialtools.terrain_tools as terrain_tools
import time
import datetime
from random import shuffle
import netCDF4 as nc
from scipy.io import netcdf as scipy_nc
import time
from dateutil.relativedelta import relativedelta
#from rpy2.robjects import r,FloatVector
from osgeo import ogr,osr
import gc
#import sparse
from scipy.sparse import csr_matrix, csc_matrix, find, hstack
import psutil
import rasterio
import fiona
from numba import jit
import json
import xarray as xr
import pandas as pd
import copy
import glob
mb = 1024*1024

def assemble_domain(comm,jfile):
    
 #define MPI parameters
 rank = comm.Get_rank()
 size = comm.Get_size()
    
 #Read metadata
 metadata = json.load(open(jfile))

 #Create shapefile
 if rank == 0:
  os.system('rm -rf %s' % metadata['output_dir'])
  domain_decomposition(metadata)
 comm.Barrier()

 #Correct and finalize the domain decomposition
 correct_domain_decomposition(comm,metadata)
 comm.Barrier()

 return

def preprocess_datasets(comm,jfile):
    
 #define MPI parameters
 rank = comm.Get_rank()
 size = comm.Get_size()
    
 #Read metadata
 metadata = json.load(open(jfile))

 #Read in the domain summary database
 pck_file = '%s/shp/domain_database.pck' % metadata['output_dir']
 mpdb = pickle.load(open(pck_file,'rb'))
 mprange = range(len(mpdb))
 for imp in mprange[rank::size]:

  print("Rank:%d, Macroscale polygon:%s" % (rank,imp),flush=True)

  mpid = mpdb[imp]['mpid']

  #Define the macroscale polygon directory
  mpdir = '%s/mps/%d' % (metadata['output_dir'],mpid)
  workspace = mpdir

  #Define the log
  log = '%s/log.txt' % mpdir

  #Prepare the workspace for the sub-domain
  os.system('rm -rf %s' % mpdir)
  os.system('mkdir -p %s' % mpdir)
    
  #Create the mask
  print(rank,'Preparing macroscale polygon mask',time.ctime(),mpid,flush=True)
  Create_Mask(mpdb[imp],mpdir,metadata,mpid,log)

  #Create land cover product
  print(rank,'Preparing land cover data',time.ctime(),mpid,flush=True)
  Extract_Land_Cover_CCI_Orchidee(mpdb,workspace,metadata,mpid,log)
    
  #Terrain analysis
  print(rank,'Preparing elev products',time.ctime(),mpid,flush=True)
  Terrain_Analysis(mpdb,workspace,metadata,mpid,log)

  #Create soil product
  print(rank,'Preparing the soil data',time.ctime(),mpid,flush=True)
  Extract_Soils(mpdb,workspace,metadata,mpid,log)
    
  #Clean up directory
  os.system('rm -rf %s/CCI_orchidee' % workspace)

 return

def compute_sfts(comm,jfile):
    
 #define MPI parameters
 rank = comm.Get_rank()
 size = comm.Get_size()
    
 #Read metadata
 metadata = json.load(open(jfile))
 bins = {}
 var_types = {}
 vars = list(metadata['histogram_info'].keys())
 for var in vars:
  bins[var] = metadata['histogram_info'][var]['bins']
  var_types[var] = metadata['histogram_info'][var]['type']

 #Read in the domain summary database
 pck_file = '%s/shp/domain_database.pck' % metadata['output_dir']
 mpdb = pickle.load(open(pck_file,'rb'))
 mprange = range(len(mpdb))
 for imp in mprange[rank::size]:
    
  print("Rank:%d, Macroscale polygon:%s" % (rank,imp),flush=True)

  mpid = mpdb[imp]['mpid']

  #Define the macroscale polygon directory
  mpdir = '%s/mps/%d' % (metadata['output_dir'],mpid)
    
  #Read in and curate covariates
  db = read_and_curate_covariates(mpdir,mpid)

  #Calculate additional derived variables
  db = assemble_additional_covariates(db,mpdir)

  #Apply binning to assemble n-d histogram
  H = compute_histogram(db,vars,bins,var_types)

  #Compute SFT database
  sft_db = compute_sft_database(H,bins,vars,var_types)

  #Assign sfts to fine-scale pixels
  sfts_map = assign_sfts(sft_db,db,vars,bins,var_types)
    
  #Save sft database and corresponding map
  write_sft_database_and_map(mpdir,sfts_map,sft_db)
    
 return

def write_file(metadata,lats,lons,nlat,nlon,bins,nsft):
    
 #Initialize empty array
 sftfct = -9999*np.ones((1,nsft,nlat,nlon)) 
 sfts = np.arange(1,nsft+1)
 #Process and place all the temporary groups of databases
 workspace = '%s/workspace' % metadata['output_dir']
 files = glob.glob('%s/*.pck' % workspace)
 for file in files:
  odb = pickle.load(open(file,'rb'))
  #for i in range(odb['ilat'].size):
  if odb['ilat'].size >= 1:sftfct[0,:,odb['ilat'],odb['ilon']] = odb['sftfct'][:]
  #elif odb['ilat'].size>2:sftfct[0,:,odb['ilat'][0],odb['ilon'][0]] = odb['sftfct'][:]
 #Prepare xarray dataset
 time_counter = pd.date_range("%4d-01-01" % metadata['preprocessing_year'], periods=1)
 ds = xr.Dataset(
        data_vars=dict(
            sftfct=(["time_counter","sft","lat","lon"],sftfct.astype(np.float32),
                    dict(long_name="SFT fractions",name="sfcfct",units="-",_FillValue=-9999)),
            sft=(["sft"],sfts.astype(np.int32),
                    dict(long_name="Surface types",units="-",axis="Z",_FillValue=-9999)),
            lon=(["lon"],lons.astype(np.float32),
                    dict(long_name="Longitude",units="degrees_east",axis="X",_FillValue=-9999)),
            lat=(["lat"],lats.astype(np.float32),
                    dict(long_name="Latitude",units="degrees_north",axis="Y",_FillValue=-9999)),
        ),
        coords=dict(
            time_counter=time_counter,
        ),
        attrs=dict(description=""))
 for var in bins:
        if var not in ['pft',]:
            for stat in ['min','max']:
                data = bins[var][stat]
                long_name = "%s_%s" % (var,stat)
                ds["%s_%s" % (var,stat)] = (["sft"],data.astype(np.float32),
                        dict(long_name=long_name,units="-",_FillValue=-9999))
        else:
            data = bins[var]['min']
            long_name = "%s" % (var,)
            ds["%s" % (var,)] = (["sft"],data.astype(np.int32),
                    dict(long_name=long_name,units="-",_FillValue=-9999))
 ofile = '%s/SFTmap_%04d.nc' % (metadata['output_dir'],metadata['preprocessing_year'])
 os.system('rm -f %s' % ofile)
 ds.coords['lon'] = (ds.coords['lon'] + 360) % 360
 ds = ds.sortby(ds.lon)
 ds.to_netcdf(ofile)

 return

def prepare_netcdf_file(comm,jfile):
  
 rank = comm.Get_rank()
 size = comm.Get_size()
 #Read metadata
 metadata = json.load(open(jfile))
 #Assemble parameters to assemble the output dataset
 minlat = metadata['domain']['bbox']['minlat'] + metadata['domain']['res']/2.0
 maxlat = metadata['domain']['bbox']['maxlat'] - metadata['domain']['res']/2.0
 minlon = metadata['domain']['bbox']['minlon'] + metadata['domain']['res']/2.0
 maxlon = metadata['domain']['bbox']['maxlon'] - metadata['domain']['res']/2.0
 nlat = int((maxlat-minlat)/metadata['domain']['res']+1)
 nlon = int((maxlon-minlon)/metadata['domain']['res']+1)
 lats = np.linspace(minlat,maxlat,nlat)
 lons = np.linspace(minlon,maxlon,nlon)
 workspace = '%s/workspace' % metadata['output_dir']
 if rank == 0:
  #Create temporary workspace
  os.system('rm -rf %s' % workspace)
  os.system('mkdir -p %s' % workspace)
 comm.Barrier()
 #Have each rank process and write the info
 #Read in the domain summary database
 pck_file = '%s/shp/domain_database.pck' % metadata['output_dir']
 mpdb = pickle.load(open(pck_file,'rb'))
 #Read in and place all the sft databases
 mprange = range(len(mpdb))
 odb = {'ilat':[],'ilon':[],'sftfct':[]}
 for imp in mprange[rank::size]:
  mpid = mpdb[imp]['mpid']
  lat = (mpdb[imp]['bbox']['minlat'] + mpdb[imp]['bbox']['maxlat'])/2
  lon = (mpdb[imp]['bbox']['minlon'] + mpdb[imp]['bbox']['maxlon'])/2
  ilat = np.argmin(np.abs(lats-lat))
  ilon = np.argmin(np.abs(lons-lon))
  pck_file = '%s/mps/%d/sfts_db.pck' % (metadata['output_dir'],mpid)
  sft_db = pickle.load(open(pck_file,'rb'))
  odb['ilat'].append(ilat)
  odb['ilon'].append(ilon)
  odb['sftfct'].append(sft_db['fct'][:])
  if imp == 0:
   #Save info for netcdf file creation
   bins = sft_db['bins']
   nsft = sft_db['sft'].size
 for var in odb:
    odb[var] = np.array(odb[var])
 #Write out temporary database
 pickle.dump(odb,open('%s/%d.pck' % (workspace,rank),'wb'),pickle.HIGHEST_PROTOCOL)
 comm.Barrier()

 if rank == 0:
  write_file(metadata,lats,lons,nlat,nlon,bins,nsft)
  #Remove temporary workspace
  os.system('rm -rf %s' % workspace)
 comm.Barrier()
    
 return

def write_sft_database_and_map(mpdir,sfts_map,sft_db):

 #Learn the metadata from the mask
 ifile = '%s/mask.tif' % mpdir
 profile = rasterio.open(ifile).profile
 profile.update(dtype=rasterio.int32)
 
 #Output the tif file
 ofile = '%s/sfts.tif' % mpdir
 fp = rasterio.open(ofile,'w',**profile)
 fp.write(sfts_map.astype(rasterio.int32),1)
 fp.close()

 #Pickle the sfts database
 pickle.dump(sft_db,open('%s/sfts_db.pck' % mpdir,'wb'),pickle.HIGHEST_PROTOCOL)

 return

def read_and_curate_covariates(mpdir,mpid):
 #read in data
 db = {}
 db['mask'] = rasterio.open('%s/mask.tif' % mpdir).read(1)
 db['elev'] = rasterio.open('%s/elev.tif' % mpdir).read(1)
 db['clay'] = rasterio.open('%s/clay.tif' % mpdir).read(1)
 db['sand'] = rasterio.open('%s/sand.tif' % mpdir).read(1)
 db['pft'] = rasterio.open('%s/pft.tif' % mpdir).read(1)
 #check all againt mask
 m = db['mask'] != mpid
 for var in ['elev','clay','pft','sand']:
    db[var][m] = -9999
 #curate elev, sand, and clay
 for var in ['elev','clay','sand']:
    if np.sum(db[var] != -9999) > 0:
        db[var][db[var]==-9999] = np.mean(db[var][db[var]!=-9999])
    else:
        db[var][:] = 0.0
 #check all against pft
 m = db['pft'] == -9999
 for var in ['elev','clay']:
    db[var][m] = -9999
 return db

def assemble_additional_covariates(db,mpdir):
 #Terrain attributes (slope and aspect)
 mask_object = gdal_tools.read_data('%s/mask.tif' % mpdir)
 terrain_tools.calculate_area(mask_object)
 res = np.mean(mask_object.area**0.5) 
 #Calculate slope and aspect
 res_array = np.copy(db['elev'])
 res_array[:] = res
 (slope,aspect) = terrain_tools.ttf.calculate_slope_and_aspect(db['elev'],res_array,res_array)
 db['slope'] = np.copy(slope)
 db['aspect_ew'] = np.sin(aspect)
 db['aspect_sn'] = np.cos(aspect)
 #add ew,sn
 (ew,sn) = np.meshgrid(np.linspace(0,1,db['mask'].shape[0]),np.linspace(0,1,db['mask'].shape[1]))
 db['ew'] = ew
 db['sn'] = sn
 #additional correction on all new variables
 for var in ['slope','aspect_sn','aspect_ew','sn','ew']:
    db[var][db['pft']==-9999]=-9999
 return db

def compute_histogram(data,vars,bins,var_types):
    #Assemble bins for histogram
    bins2 = []
    for var in vars:
        if var_types[var] == 'class':
            tmp = copy.deepcopy(bins[var])
            tmp.append(np.max(tmp)+1)
            bins2.append(tmp)
        else:
            bins2.append(bins[var])
    #Assemble X
    X = []
    for var in vars:
        m = data[var] != -9999
        X.append(data[var][m])
    X = np.array(X).T
    #Compute histogram
    (H,edges) = np.histogramdd(X,bins=bins2)
    #Normalize for fractions
    H = H/np.sum(H)
    return H

def compute_sft_database(H,bins,vars,var_types):
    sft_db = {'sft':np.arange(1,H.size+1),
              'bins':{},'fct':np.zeros(H.size)}
    for var in vars:
        sft_db['bins'][var] = {'min':np.zeros(H.size),'max':np.zeros(H.size)}
    count = 0
    for index, fct in np.ndenumerate(H):
        sft_db['fct'][count] = fct
        for i in range(len(index)):
            var = vars[i]
            tmp = copy.deepcopy(bins[var])
            if var_types[var] == 'class':tmp.append(np.max(tmp)+1)
            sft_db['bins'][var]['min'][count] = tmp[index[i]]
            sft_db['bins'][var]['max'][count] = tmp[index[i]+1]
        count += 1
    return sft_db

def assign_sfts(sft_db,db,vars,bins,var_types):
    sfts_map = -9999*np.ones(db['mask'].shape).astype(np.int32)
    m = np.zeros(db['mask'].shape).astype(np.bool_)
    for i in range(sft_db['sft'].size):
        if sft_db['fct'][i] == 0.0:continue
        for var in vars:
            vmin = sft_db['bins'][var]['min'][i]
            vmax = sft_db['bins'][var]['max'][i]
            if var_types[var] == 'class':vmax = vmax + 1 #for classes we don't want the center but we want the first; need to add an artificial class at the end
            if vmax == np.max(bins[var]):
                if var == vars[0]:
                    m = (db[var] >= sft_db['bins'][var]['min'][i]) & (db[var] <= sft_db['bins'][var]['max'][i])
                else:
                    m = m & (db[var] >= sft_db['bins'][var]['min'][i]) & (db[var] <= sft_db['bins'][var]['max'][i])
            else:
                if var == vars[0]:
                    m = (db[var] >= sft_db['bins'][var]['min'][i]) & (db[var] < sft_db['bins'][var]['max'][i])
                else:
                    m = m & (db[var] >= sft_db['bins'][var]['min'][i]) & (db[var] < sft_db['bins'][var]['max'][i])
        sfts_map[m] = sft_db['sft'][i]            
    return sfts_map

def domain_decomposition(md):

 #Make new lat/lon domain
 create_domain_shapefile(md)
 #Create summary database
 summarize_domain_decompisition(md)

 return

def create_domain_shapefile(md):

 #Extract parameters
 minlat = md['domain']['bbox']['minlat']
 maxlat = md['domain']['bbox']['maxlat']
 minlon = md['domain']['bbox']['minlon']
 maxlon = md['domain']['bbox']['maxlon']
 res = md['domain']['res']
 sdir = '%s/shp' % md['output_dir']
 os.system('mkdir -p %s' % sdir)

 #Create the shapefile
 driver = ogr.GetDriverByName("ESRI Shapefile")
 ds = driver.CreateDataSource("%s/domain.shp" % sdir)
 srs = osr.SpatialReference()
 srs.ImportFromEPSG(4326)
 layer = ds.CreateLayer("grid", srs, ogr.wkbPolygon)
 layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
 layer.CreateField(ogr.FieldDefn("X", ogr.OFTInteger))
 layer.CreateField(ogr.FieldDefn("Y", ogr.OFTInteger))

 #Define lats,lons
 nlat = int(np.round((maxlat-minlat)/res)) + 1
 nlon = int(np.round((maxlon-minlon)/res)) + 1
 lats = np.linspace(minlat,maxlat,nlat)
 lons = np.linspace(minlon,maxlon,nlon)

 #Iterate through each cell
 cid = 1
 for ilat in range(lats.size-1):
  for ilon in range(lons.size-1):
   #Construct list of points
   mpoint = ogr.Geometry(ogr.wkbMultiPoint)
   point = ogr.Geometry(ogr.wkbPoint)
   iss = [0,0,1,1,0]
   jss = [0,1,1,0,0]
   for k in range(len(iss)):
    i = iss[k]
    j = jss[k]
    point.AddPoint(lons[ilon+j],lats[ilat+i])
    mpoint.AddGeometry(point)
   poly = mpoint.ConvexHull()
   #Add to the info
   # create the feature
   feature = ogr.Feature(layer.GetLayerDefn())
   #Set the ID
   feature.SetField("ID",cid)
   #Set the i id
   feature.SetField("X",ilat)
   #Set the j id
   feature.SetField("Y",ilon)
   #Set the feature geometry using the point
   feature.SetGeometry(poly)
   #Create the feature in the layer (shapefile)
   layer.CreateFeature(feature)
   #Destroy the feature to free resources
   feature.Destroy()
   #Update cid
   cid += 1

 #Destroy the data source to free resources
 ds.Destroy()

 return md

def summarize_domain_decompisition(md):

 sdir = '%s/shp' % md['output_dir']
 ifile = "%s/domain.shp" % sdir

 #Prepare the domain directory
 mps_dir = '%s/mps' % md['output_dir']
 os.system('mkdir -p %s' % mps_dir)

 #Open access to the database
 driver = ogr.GetDriverByName("ESRI Shapefile")
 ds = driver.Open(ifile,0)

 #Iterate through each feature getting the necessary info
 layer = ds.GetLayer()
 output = []
 for feature in layer:
   info = {}
   bbox = feature.GetGeometryRef().GetEnvelope()
   info['mpid'] = feature.GetField('ID')
   info['bbox'] =  {'minlat':bbox[2],'minlon':bbox[0],'maxlat':bbox[3],'maxlon':bbox[1]}
   output.append(info)

 #Pickle the database
 pickle.dump(output,open('%s/domain_database.pck' % sdir,'wb'),pickle.HIGHEST_PROTOCOL)

 return

def Create_Mask(mpdb,workspace,metadata,mpid,log):

 #Define parameters
 shp_in = '%s/shp' % metadata['output_dir']
 cistr = "ID"
 lstr = "domain"
 bbox =  mpdb['bbox']
 res = metadata['preprocessing_resolution']
 
 #Define the files
 mask_file = '%s/mask.tif' % workspace
 tmp_file = '%s/tmp.tif' % workspace
 
 #Rasterize the area
 buff = 0.0
 
 print(' buffer size:',buff,' mpid:',mpid,flush=True) 
 minx = bbox['minlon']-buff
 miny = bbox['minlat']-buff
 maxx = bbox['maxlon']+buff
 maxy = bbox['maxlat']+buff
 cache = int(psutil.virtual_memory().available*0.7/mb)
 print(minx,maxx,miny,maxy)

 #Subset region
 shp_out = '%s/domain' % workspace
 os.system('ogr2ogr -spat %.16f %.16f %.16f %.16f %s %s >> %s 2>&1' % (minx,miny,maxx,maxy,shp_out,shp_in,log))

 #Rasterize
 os.system("gdal_rasterize -at -ot Float64 --config GDAL_CACHEMAX %i -a_nodata -9999 -init -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f -a %s -l %s %s %s >> %s 2>&1" % (cache, res,res,minx,miny,maxx,maxy,cistr,lstr,shp_out,mask_file,log))
    
 #Update to completely will out region (HACK; only for 0 buffer application)
 fp = rasterio.open(mask_file)
 data = fp.read(1)
 data[:] = mpid
 profile = fp.profile
 fp.close()
 fp = rasterio.open(mask_file,'w',**profile)
 fp.write(data,1)
 fp.close()

 #del data
 gc.collect()

 return

def Terrain_Analysis(cdb,workspace,metadata,icatch,log):

 #0. Get the parameters
 md = gdal_tools.retrieve_metadata('%s/mask.tif' % workspace)
 minx = md['minx']
 miny = md['miny']
 maxx = md['maxx']
 maxy = md['maxy']
 res = abs(md['resx'])
 elev_region = metadata['input_files']['elev']
 lproj = md['proj4']

 #1. Cutout the region of interest
 elev_latlon_file = '%s/elev.tif' % workspace
 cache = int(psutil.virtual_memory().available*0.7/mb)
 os.system('gdalwarp -t_srs \'%s\' -dstnodata -9999.0 -te %.16f %.16f %.16f %.16f --config GDAL_CACHEMAX %i  %s %s >> %s 2>&1' % (lproj,minx,miny,maxx,maxy,cache,elev_region,elev_latlon_file,log))

 data = gdal_tools.read_raster(elev_latlon_file)
 metadata = gdal_tools.retrieve_metadata(elev_latlon_file)
 metadata['nodata'] = -9999.0
 data[data == -32768] = -9999.0
 gdal_tools.write_raster(elev_latlon_file,metadata,data)

 del data
 gc.collect()
 
 return

def Extract_Land_Cover_CCI_Orchidee(mpdb,workspace,metadata,mpid,log):

 #0. Get the parameters
 md = gdal_tools.retrieve_metadata('%s/mask.tif' % workspace)
 minx = md['minx']
 miny = md['miny']
 maxx = md['maxx']
 maxy = md['maxy']
 res = abs(md['resx'])
 preprocessing_year = metadata['preprocessing_year']
 gpft_file = metadata['input_files']['gptfs_file'] % preprocessing_year
 kg_file = metadata['input_files']['kg_file']
 still_file = metadata['input_files']['still_file']
 hydrolakes_file = metadata['input_files']['hydrolakes_file']
 luh_file = metadata['input_files']['luh_file']
 lproj = md['proj4']

 #1.1 Extract CCI generic PFTs
 data = []
 cdir = '%s/CCI_orchidee' % workspace
 os.system('mkdir -p %s' % cdir)
 gpfts = ['BARE','BUILT','GRASS-MAN','GRASS-NAT','SHRUBS-BD','SHRUBS-BE',
          'SHRUBS-ND','SHRUBS-NE','WATER_INLAND','SNOWICE','TREES-BD','TREES-BE',
          'TREES-ND','TREES-NE','WATER_OCEAN','LAND','WATER']
 for gpft in gpfts:
  fin = 'NETCDF:"%s":%s' % (gpft_file,gpft)
  fout = '%s/GPFT_%s.tif' % (cdir,gpft)
  cache = int(psutil.virtual_memory().available*0.7/mb)
  os.system('gdalwarp -t_srs \'%s\' -ot Int16 -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f --config GDAL_CACHEMAX %i %s %s >> %s 2>&1' % (lproj,res,res,minx,miny,maxx,maxy,cache,fin,fout,log))
  if gpft in ['LAND','WATER']:continue
  data.append(rasterio.open(fout).read(1).astype(np.float32))
 data = np.array(data)
 data[data!=-9999]=data[data!=-9999]/100.0
 #print('data',np.unique(data))

 #clean up at -180/180 spurious artifacts
 if ((minx > 178.0) | (minx < -178.0)):
    idx1 = gpfts.index('GRASS-NAT')
    idx2 = gpfts.index('WATER_INLAND')
    #print(np.unique(data[idx1,:,:]),np.unique(data[idx2,:,:]))
    m = (data[idx1,:,:]==0.14) & (data[idx2,:,:]==0.86)
    data[idx1,:,:][m] = -9999
    data[idx2,:,:][m] = -9999
 #Remove south of -60
 if (maxy < -60.0):
    data[:,:,:] = -9999
 #1.2 Sample per 90 meter pixel to account for variability from CCI
 pft_map = random_disaggregation_landcover(data)
    
 #1.3 Randomly reassign urban to 80% bare soil and 20% natural grass (lack of urban constraint)
 np.random.seed(1)
 random_array = np.random.uniform(low=0,high=1,size=(pft_map.shape))
 pft_map[(pft_map == gpfts.index('BUILT')) & (random_array < 0.8)] = gpfts.index('BARE')
 pft_map[(pft_map == gpfts.index('BUILT')) & (random_array >= 0.8)] = gpfts.index('GRASS-NAT')

 #3.1.Extract Koppen Geiger data
 fin = kg_file
 fout = '%s/kg.nc' % cdir
 cache = int(psutil.virtual_memory().available*0.7/mb)
 os.system('gdalwarp -ot Int32 -of netcdf -t_srs \'%s\' -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f --config GDAL_CACHEMAX %i %s %s >> %s 2>&1' % (lproj,res,res,minx,miny,maxx,maxy,cache,fin,fout,log))

 #2.2 Reduce Koppen Geiger classes
 fp = nc.Dataset(fout)
 data = []
 for band in range(1,32):
     string = 'Band%d' % band
     data.append(fp[string][:].astype(np.int32))
 data = np.array(data)
 #1.2 Sample per 90 meter pixel to account for variability from KG
 #tmp = random_disaggregation_landcover(data)
 tmp = np.argmax(data,axis=0)
 #Map to the subset
 class_names = ["Af", "Am", "As", "Aw", "BSh", "BSk", "BWh", "BWk", "Cfa", 
     "Cfb", "Cfc", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Dfa", "Dfb", 
     "Dfc", "Dfd", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", 
     "EF", "ET"]
 mapping = {0:0,1:0,2:0,3:0,4:1,5:2,6:1,7:2,8:3,9:4,10:4,11:3,12:4,13:4,14:3,15:4,16:4,17:4,18:5,19:6,20:6,21:5,22:5,
           23:6,24:6,25:5,26:5,27:6,28:6,29:6,30:6}
 uclasses = np.unique(tmp)
 #uclasses = uclasses[uclasses!=-9999]
 data = np.copy(tmp)
 data[:] = -9999
 for uclass in uclasses:
     data[tmp == uclass] = mapping[uclass]
 kg_map = np.copy(data).astype(np.int32)
 kg_classes = ['tropical','arid_warm','arid_cool','temp_warm','temp_cool','boreal_warm','boreal_cool']   

 #3.1 Extract Still C4 fraction data
 fin = still_file
 fout = '%s/still.nc' % cdir
 cache = int(psutil.virtual_memory().available*0.7/mb)
 os.system('gdalwarp -ot Float32 -of netcdf -t_srs \'%s\' -dstnodata -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f --config GDAL_CACHEMAX %i %s %s >> %s 2>&1' % (lproj,res,res,minx,miny,maxx,maxy,cache,fin,fout,log))

 #3.2 Process Still C4 fraction data
 fp = nc.Dataset(fout)
 data_still = fp['Band1'][:]
 np.random.seed(1)
 random_array = np.random.uniform(low=0,high=1,size=(data_still.shape))
 c4_coverage = np.zeros(data_still.shape).astype(np.int32)
 c4_coverage[random_array < data_still] = 1
    
 #4.1 Extract HydroLakes data
 shp_in = hydrolakes_file
 shp_out = '%s/lakes_latlon.shp' % cdir
 os.system('ogr2ogr -spat %.16f %.16f %.16f %.16f %s %s >> %s 2>&1' % (minx,miny,maxx,maxy,shp_out,shp_in,log))
 #Rasterize
 hydrolakes_latlon_file = '%s/hydrolakes.tif' % cdir
 os.system("gdal_rasterize -at -ot Float64 --config GDAL_CACHEMAX %i -a_nodata -9999 -init -9999 -tr %.16f %.16f -te %.16f %.16f %.16f %.16f -burn 1 %s %s >> %s 2>&1" % (cache,res,res,minx,miny,maxx,maxy,shp_out,hydrolakes_latlon_file,log))
 data_hydrolakes = rasterio.open(hydrolakes_latlon_file).read(1)

 #5.1 Extract LUH data
 fin = luh_file
 fout = '%s/luh.nc' % cdir
 gridfile = '%s/gridfile.txt' % cdir
 fp = open(gridfile,'w')
 nlat = data_hydrolakes.shape[0]
 nlon = data_hydrolakes.shape[1]
 fp.write('gridtype  = lonlat\n')
 fp.write('xsize     = %d\n' % nlon)
 fp.write('ysize     = %d\n' % nlat)
 fp.write('xname     = lon\n')
 fp.write('xlongname = "longitude"\n')
 fp.write('xunits    = "degrees_east"\n')
 fp.write('yname     = lat\n')
 fp.write('ylongname = "latitude"\n')
 fp.write('yunits    = "degrees_north"\n')
 fp.write('xfirst    = %.16f\n' % minx)
 fp.write('xinc      = %.16f\n' % res)
 fp.write('yfirst    = %.16f\n' % miny)
 fp.write('yinc      = %.16f' % res)
 fp.close()
 iyear = list(range(1992,2023)).index(preprocessing_year) + 1
 os.system('cdo -seltimestep,%d -selname,c4ann,c4per -remapnn,%s %s %s >> %s 2>&1' % (iyear,gridfile,fin,fout,log))
 fp = nc.Dataset(fout)
 c4ann = fp['c4ann'][0,:,:]
 c4per = fp['c4per'][0,:,:]
 fp.close()
 data_luh = np.ma.getdata(c4ann + c4per)
 data_luh[data_luh>1000]=0
 np.random.seed(1)
 random_array = np.random.uniform(low=0,high=1,size=(data_luh.shape))
 c4_coverage_luh = np.zeros(data_luh.shape).astype(np.int32)
 c4_coverage_luh[random_array < data_luh] = 1

 #6.1 Convert Generic PFTs to Orchidee PFTs leveraging the auxiliary KG and Still data
 pfts = np.zeros(pft_map.shape)
 pfts[:] = -9999
 #pft1 -> bare soil
 m = pft_map == gpfts.index('BARE')
 pfts[m] = 1
 m = pft_map == gpfts.index('WATER_INLAND')
 pfts[m] = 17
 m = (pft_map == gpfts.index('WATER_OCEAN')) & (data_hydrolakes == 1) #Constrain using hydrolakes to include the big lakes (classified as ocean in CCI)
 pfts[m] = 17
 m = pft_map == gpfts.index('SNOWICE')
 pfts[m] = 16
 m = pft_map == gpfts.index('BUILT')
 pfts[m] = 1
 #m = pft_map == -9999
 #pfts[m] = 1
 #pft2 -> tropical evergreen
 for kg_class in ['tropical','arid_warm']:
  m = ((pft_map == gpfts.index('TREES-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 2
  m = ((pft_map == gpfts.index('SHRUBS-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 2
 #pft3 -> tropical raingreen
 for kg_class in ['tropical','arid_warm']:
  m = ((pft_map == gpfts.index('TREES-BD')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 3
  m = ((pft_map == gpfts.index('TREES-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 3
  m = ((pft_map == gpfts.index('SHRUBS-BD')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 3
  m = ((pft_map == gpfts.index('SHRUBS-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 3
 #pft4 -> temperate needleleaf evergreen
 for kg_class in ['tropical','arid_warm','arid_cool','temp_warm','temp_cool','boreal_warm']:
  m = ((pft_map == gpfts.index('TREES-NE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 4
  m = ((pft_map == gpfts.index('SHRUBS-NE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 4
 #pft5 -> temperate broadleaf evergreen
 for kg_class in ['arid_cool','temp_warm','temp_cool']:
  m = ((pft_map == gpfts.index('TREES-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 5
  m = ((pft_map == gpfts.index('SHRUBS-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 5
 #pft6 -> temperate broadleaf summergreen
 for kg_class in ['arid_cool','temp_warm','temp_cool','boreal_warm']:
  m = ((pft_map == gpfts.index('TREES-BD')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 6
  m = ((pft_map == gpfts.index('SHRUBS-BD')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 6
 for kg_class in ['arid_cool','temp_warm']:
  m = ((pft_map == gpfts.index('TREES-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 6
  m = ((pft_map == gpfts.index('SHRUBS-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 6
 #pft7 -> boreal needleaf evergreen
 for kg_class in ['boreal_warm','boreal_cool']:
  m = ((pft_map == gpfts.index('TREES-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 7
  m = ((pft_map == gpfts.index('SHRUBS-BE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 7
 for kg_class in ['boreal_cool',]:
  m = ((pft_map == gpfts.index('TREES-NE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 7
  m = ((pft_map == gpfts.index('SHRUBS-NE')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 7
 #pft8 -> boreal broadleaf summergreen
 m = ((pft_map == gpfts.index('TREES-BD')) & (kg_map == kg_classes.index('boreal_cool')))
 pfts[m] = 8
 m = ((pft_map == gpfts.index('SHRUBS-BD')) & (kg_map == kg_classes.index('boreal_cool')))
 pfts[m] = 8
 #pft9 -> boreal needleaf deciduous
 for kg_class in ['temp_cool','boreal_warm','boreal_cool']:
  m = ((pft_map == gpfts.index('TREES-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 9
  m = ((pft_map == gpfts.index('SHRUBS-ND')) & (kg_map == kg_classes.index(kg_class)))
  pfts[m] = 9
 #pft10 -> temperate natural grassland (c3)
 for kg_class in ['arid_cool','temp_warm','temp_cool']:
  m = ((pft_map == gpfts.index('GRASS-NAT')) \
       & (kg_map == kg_classes.index(kg_class))\
       & (c4_coverage == 0))
  pfts[m] = 10
 #pft11 -> natural grassland (c4)
 m = ((pft_map == gpfts.index('GRASS-NAT')) & (c4_coverage == 1))
 pfts[m] = 11
 #pft12 -> crops (c3)
 m = ((pft_map == gpfts.index('GRASS-MAN')) & (c4_coverage_luh == 0))
 pfts[m] = 12
 #pft13 -> crops (c4)
 m = ((pft_map == gpfts.index('GRASS-MAN')) & (c4_coverage_luh == 1))
 pfts[m] = 13
 #pft14 -> tropical natural grassland (c3)
 for kg_class in ['tropical','arid_warm']:
  m = ((pft_map == gpfts.index('GRASS-NAT')) \
         & (kg_map == kg_classes.index(kg_class))\
         & (c4_coverage == 0))
  pfts[m] = 14
 #pft15 -> boreal natural grassland (c3)
 for kg_class in ['boreal_warm','boreal_cool']:
  m = ((pft_map == gpfts.index('GRASS-NAT')) \
         & (kg_map == kg_classes.index(kg_class))\
         & (c4_coverage == 0))
  pfts[m] = 15
 
 #6.2. Export to tif
 orchidee_pfts_file = '%s/pft.tif' % workspace
 md = gdal_tools.retrieve_metadata(hydrolakes_latlon_file)
 md['nodata'] = -9999.0
 gdal_tools.write_raster(orchidee_pfts_file,md,pfts)
 tmp = '%s/pft_ea2.tif' % workspace
 cache = int(psutil.virtual_memory().available*0.7/mb)
 os.system('gdal_translate --config GDAL_CACHEMAX %i %s %s >> %s 2>&1' % (cache,orchidee_pfts_file,tmp,log))
 os.system('mv %s %s && rm -rf %s >> %s 2>&1' % (tmp,orchidee_pfts_file,tmp,log))

 del data, tmp
 gc.collect()

 return

def random_disaggregation_landcover(probs):
    cumsum_probs = np.cumsum(probs,axis=0)
    np.random.seed(1)
    pft_map = np.zeros((probs.shape[1],probs.shape[2]))
    pft_map[:] = -9999
    random_array = np.random.uniform(low=0,high=1,size=(pft_map.shape))
    classes = np.arange(probs.shape[0])
    return random_disaggregation_landcover_workhorse(cumsum_probs,pft_map,random_array,classes)

@jit(nopython=True)
def random_disaggregation_landcover_workhorse(cumsum_probs,pft_map,random_array,classes):
    for i in range(pft_map.shape[0]):
        for j in range(pft_map.shape[1]):
            for k in range(classes.size):
                if (k == 0):
                    if (random_array[i,j] < cumsum_probs[k,i,j]):
                        pft_map[i,j] = k
                        break
                elif (k == classes.size-1):
                    if (random_array[i,j] < cumsum_probs[k,i,j]):
                        pft_map[i,j] = k
                        break
                else: 
                    if (random_array[i,j] < cumsum_probs[k,i,j]) & (random_array[i,j] >= cumsum_probs[k-1,i,j]):
                        pft_map[i,j] = k
                        break
    return pft_map

def Extract_Soils(cdb,workspace,metadata,icatch,log):

 #0. Get the parameters
 md = gdal_tools.retrieve_metadata('%s/mask.tif' % workspace)
 minx = md['minx']
 miny = md['miny']
 maxx = md['maxx']
 maxy = md['maxy']
 res = abs(md['resx'])
 lproj = md['proj4']

 # SOILGRIDS
 vars = ['clay','sand','silt']
 shuffle(vars)
 properties = {}
 for var in vars:
  file_in = metadata['input_files'][var]
  file_out = '%s/%s.tif' % (workspace,var)
  cache = int(psutil.virtual_memory().available*0.7/mb)
  os.system('gdalwarp -t_srs \'%s\' -dstnodata -9999 -r bilinear -tr %.16f %.16f -te %.16f %.16f %.16f %.16f --config GDAL_CACHEMAX %i %s %s >> %s 2>&1' % (lproj,res,res,minx,miny,maxx,maxy,cache,file_in,file_out,log))
  #Get the data
  properties[var] = gdal_tools.read_raster(file_out)
  if var in ['clay','sand','silt']:
   properties[var][(properties[var] == 255)] = -9999.0
  if var in ['om',]:
   badvals = (properties['om'] == -9999.0)
   properties['om'] = 100*(properties['om']/1000.0)  # g/kg -> g/g -> %w
   # soigrids uses organic carbon, which is OC = 0.58*OM
   properties['om'] = 1.724*properties['om']
   properties['om'][badvals] = -9999.0
   
 # Write out the data
 mask = gdal_tools.read_raster('%s/mask.tif' % workspace)
 badvals = ((properties['clay']==-9999.0) | (properties['sand']==-9999.0)) | ((properties['silt']==-9999.0))
 for var in ['clay','sand','silt']:
  file_out = '%s/%s.tif' % (workspace,var)
  properties[var][badvals] = -9999.0
  md = gdal_tools.retrieve_metadata(file_out)
  md['nodata'] = -9999.0
  gdal_tools.write_raster(file_out,md,properties[var])

 del properties
 gc.collect()

 return

def correct_domain_decomposition(comm,metadata):

 size = comm.Get_size()
 rank = comm.Get_rank()

 #Read in the domain summary database
 pck_file = '%s/shp/domain_database.pck' % metadata['output_dir']
 mpdb = pickle.load(open(pck_file,'rb'))
 mprange = range(len(mpdb))
 odb = {}
 for imp in mprange[rank::size]:

  print("Rank:%d, Macroscale polygon:%s - Initializing" % (rank,imp),flush=True)

  mpid = mpdb[imp]['mpid']

  #Define the macroscale polygon directory
  mpdir = '%s/mps/%d' % (metadata['output_dir'],mpid)
  workspace = mpdir

  #Define the log
  log = '%s/log.txt' % mpdir

  #Prepare the workspace for the sub-domain
  os.system('rm -rf %s' % mpdir)
  os.system('mkdir -p %s' % mpdir)
  #Create the mask
  print(rank,'Preparing macroscale polygon mask',time.ctime(),mpid,flush=True)
  Create_Mask(mpdb[imp],mpdir,metadata,mpid,log)

  #Create land cover product
  print(rank,'Preparing land cover data',time.ctime(),mpid,flush=True)
  Extract_Land_Cover_CCI_Orchidee(mpdb,workspace,metadata,mpid,log)
    
  #Determine number of valid pixels
  file = '%s/mask.tif' % workspace
  mask = rasterio.open(file).read(1)
  file = '%s/pft.tif' % workspace
  pft = rasterio.open(file).read(1)
  #npx_mask = np.sum(mask == mpid)
  npx = np.sum((mask == mpid) & (pft != -9999))
  #odb[cid] = min(npx_mask,npx_pft) #only land cover decides what stays and goes
  odb[mpid] = npx
    
  #Clean up directory
  os.system('rm -rf %s/CCI_orchidee' % workspace)
    
 #Wait until all complete
 comm.Barrier()

 #Broadcast and collect
 if rank == 0:
   print(rank,'Cleaning up mpid map',time.ctime(),flush=True)
   for i in range(1,size):
    odb2 = comm.recv(source=i, tag=11)
    for key in odb2:odb[key] = odb2[key]
 else:
    comm.send(odb,dest=0, tag=11)
    
 #Wait until all complete
 comm.Barrier()

 #Create new shapefile and summary
 if rank == 0:
  odb2 = {}
  count = 1
  for key in odb:
   if odb[key] > 0:#Minimum number of valid pixels in the cid of the domain to count
    odb2[key] = count
    count += 1
    
  odb = odb2
  dfile = '%s/shp/domain.shp' % metadata['output_dir']
  dfile2 = '%s/shp/tmp.shp' % metadata['output_dir']
  fp = fiona.open(dfile,'r')
  fp2 = fiona.open(dfile2,'w',crs=fp.crs,driver='ESRI Shapefile',schema=fp.schema)
  for poly in fp.values():
   ID = poly['properties']['ID']
   if ID in odb:
    poly['id'] = odb[ID]-1
    poly['properties']['ID'] = odb[ID]
    fp2.write(poly)
  fp.close()
  fp2.close()
  mdir = metadata['output_dir']
  os.system('mv %s/shp/tmp.shp %s/shp/domain.shp' % (mdir,mdir))
  os.system('mv %s/shp/tmp.shx %s/shp/domain.shx' % (mdir,mdir))
  os.system('mv %s/shp/tmp.prj %s/shp/domain.prj' % (mdir,mdir))
  os.system('mv %s/shp/tmp.cpg %s/shp/domain.cpg' % (mdir,mdir))
  os.system('mv %s/shp/tmp.dbf %s/shp/domain.dbf' % (mdir,mdir))
  #perform new summary
  summarize_domain_decompisition(metadata)

 #Wait until rank 0 completes
 comm.Barrier()

 #Remove all mpdir
 for imp in mprange[rank::size]:

  mpid = mpdb[imp]['mpid']

  #Define the macroscale polygon directory
  mpdir = '%s/mps/%d' % (metadata['output_dir'],mpid)
  os.system('rm -rf %s' % mpdir)

 comm.Barrier()

 return