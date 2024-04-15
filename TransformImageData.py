import numpy as np
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
from PIL import Image
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2 as cv 
import skimage
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from ModelFunctions.HumanSort import *

### MODIS DATA ###

# Load in raw hdf MODIS files
def MODISLoad_Raw(filepath,outpath, scale_factor = 1.2988):
    """
    ----Arguments----
    filepath -> String containing the path to the directory where the MODIS hdf files are stored
    outpath -> String containing the path to the directory where you'd like to deposit the raw images
    scale_factor -> Float representing the conversion needed to make matplotlib save figures in the proper dimension

    ---- Outputs ----
    Nothing except the for png's of resolution 2030x1354 deposited in the outpath directory

    ---- Purpose ----
    This function takes in raw MODIS data simply save the Atmosphere Corrected Reflection data in png
    format of dimension 2030X1354
    """

    modisfiles = glob.glob(filepath + '*.hdf')
    human_sort(modisfiles)
    # Set resolution factor for matplotlib
    # Plot Modis Data as raw 2030 x 1354 images
    for i in range(len(modisfiles)):
        # Get the image
        hdf = SD(modisfiles[i], SDC.READ)
        image = hdf.select('Atm_Corr_Refl')[:,:,4]
        #image = Image.fromarray(image)
        # Get the date
        yearday = modisfiles[i].split('.')[1][1:]
        hourminute = modisfiles[i].split('.')[2]
        plt.figure(figsize=(13.54*scale_factor,20.30*scale_factor),facecolor = 'None')
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(outpath+ yearday + '_' + hourminute + '_Raw.png', dpi = 100, bbox_inches='tight', pad_inches = 0)
        plt.close()

def MODISProject(datafiles, labelfiles,outpath,label=True,scale_factor=1.299):
    """
    ---- Arguments ----
    datafiles -> String containing the path to the MODIS files in hdf format
    labelfiles -> String containing the path to the masked images of MODIS raw data in png format of resolution 2030x1354
    outpath -> String containing the path to the directory in which the projected png's are to be deposited
    label -> Boolean to set depending on whether you do or do not have label images to also project
    scale_factor -> Float representing the conversion needed to make matplotlib save figures in the proper dimension

    ---- Outputs ----
    Nothing except for the raw MODIS data projected over the actual coordinates of its measure and saved 
    as a png of dimension 2030x1354

    ---- Purpose ----
    This function takes in raw MODIS data and masked images (or any image in the correct 2030x1354 dimension)
    and uses the coordinate data contained in the MODIS hdf files to project the Atmosphere Corrected Reflection
    data from MODIS over the correct lat/lon span. It then does the same projection for the label file. Since each 
    MODIS observation has a slightly different projection the data and label must be done at the same time. 

    """
    for i in range(0,len(datafiles)):
        ### First the Raw Data
        file = datafiles[i]
        hdf = SD(file, SDC.READ)
        # Read dataset.
        data2D = hdf.select('Atm_Corr_Refl')
        np.shape(data2D)
        data = data2D[:,:].astype(np.double)
        # Read geolocation dataset.
        lat = hdf.select('Latitude')
        latitude = lat[:,:]
        lon = hdf.select('Longitude')
        longitude = lon[:,:]
        if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
            longitude = np.abs(longitude)
        # Retrieve attributes.
        attrs = data2D.attributes(full=1)
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        data[data == _FillValue] = np.min(data) #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        latitude_m = latitude
        longitude_m = longitude

        # Find middle location for plotting
        lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
        lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]
        min_lon = np.min(longitude_m)
        max_lon = np.max(longitude_m)
        min_lat = np.min(latitude_m)
        max_lat = np.max(latitude_m)
        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) #lon
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


        ##FILL in NaNs with image interpolation 
        mask = data_m.mask
        datam = data_m.data 
        #standardize data to be between 0 and 1
        axis = 4
        datam_s = (datam[:,:,axis] - np.nanmin(datam[:,:,axis])) / (np.nanmax(datam[:,:,axis])-np.nanmin(datam[:,:,axis]))
        data_m2 = cv.inpaint(np.array(datam_s*255,dtype=np.uint8),np.array(mask[:,:,0],dtype=np.uint8), \
            10, cv.INPAINT_TELEA)

        # PLOTTING Data
        plt.figure(figsize=(20.30*scale_factor,13.54*scale_factor),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, np.flip(data_m2*255.,axis=0)) #flip x direction
        # Save Figure
        yearday = datafiles[i].split('.')[1][1:]
        hourminute = datafiles[i].split('.')[2]
        plt.savefig(outpath + yearday + '_' + hourminute + '_Data.png',dpi=100, bbox_inches='tight', pad_inches = 0)

        if label:
            ### Now do the Masked Label
            image = np.asarray(Image.open(labelfiles[i]))[:,:,0:3]
            scale_factor = 2030 / image.shape[0]
            temp = skimage.transform.rescale(image, scale_factor)
            temp = temp[:,:,4]
            scaled_label = np.zeros([y.shape[0],x.shape[0]])
            if (temp.shape[0] <= y.shape[0]) and (temp.shape[1] <= x.shape[0]):
                scaled_label[:temp.shape[0], :temp.shape[1]] = temp[:,:]
            elif (temp.shape[0] >= y.shape[0]) and (temp.shape[1] >= x.shape[0]):
                scaled_label = temp[:y.shape[0], :x.shape[0]]

            # PLOTTING Labels
            plt.figure(figsize=(20.30*scale_factor,13.54*scale_factor),facecolor = 'None')
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
            ax.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree(central_longitude=0))
            filled_c = ax.pcolormesh(x, y, np.flip(scaled_label*255.,axis=0))
            # Save Figure
            plt.savefig(outpath + yearday + '_' + hourminute + '_Label.png', dpi = 100, bbox_inches='tight', pad_inches = 0)    

def MODISProject_Swath(datafiles, labelfiles,outpath,label=True,scale_factor=1.299):
    """
    ---- Arguments ----
    datafiles -> String containing the path to the MODIS files in hdf format
    labelfiles -> String containing the path to the masked images of MODIS raw data in png format of resolution 2030x1354
    outpath -> String containing the path to the directory in which the projected swath png's are to be deposited
    label -> Boolean to set depending on whether you do or do not have label images to also project
    scale_factor -> Float representing the conversion needed to make matplotlib save figures in the proper dimension

    ---- Outputs ----
    Nothing except for the raw MODIS data projected over the actual coordinates of its measure and then over
    the correct shape given a spherical earth. This output is then saved as a png of whatever dimension holds
    the full swath

    ---- Purpose ----
    This function takes in raw MODIS data and masked images (or any image in the correct 2030x1354 dimension)
    and uses the coordinate data contained in the MODIS hdf files to project the Atmosphere Corrected Reflection
    data from MODIS over the correct lat/lon span and then plots it over the correct swath. It then does the same 
    projection for the label file. Since each MODIS observation has a slightly different projection the data and 
    label must be done at the same time. 

    The difference between this function and MODISProject is that this fully places the data into the correct 
    context and shows the data as it would appear on the Earth if you were the collecting Satellite. While an extra
    step that places the data into clearer context given the shape and variable dimension of the different swaths
    its advisable to use the projected images instead of these projected swath images. 

    """
    for i in range(0,len(datafiles)):
        file = datafiles[i]
        hdf = SD(file, SDC.READ)
        # Read dataset.
        data2D = hdf.select('Atm_Corr_Refl')
        np.shape(data2D)
        data = data2D[:,:].astype(np.double)
        #print(np.shape(data))
        # Read geolocation dataset.
        lat = hdf.select('Latitude')
        latitude = lat[:,:]
        lon = hdf.select('Longitude')
        longitude = lon[:,:]
        if (np.max(lon[:,:]) - np.min(lon[:,:])) > 350:
            above_indices = longitude > 0.
            below_indices = longitude < 0.
            longitude[above_indices] = (longitude[above_indices]) - 180
            longitude[below_indices] = (longitude[below_indices]) + 180
        # Retrieve attributes.
        attrs = data2D.attributes(full=1)
        aoa=attrs["add_offset"]
        add_offset = aoa[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        latitude_m = latitude
        longitude_m = longitude

        # Find middle location for plotting
        lat_m = latitude[int(latitude.shape[0]/2),int(latitude.shape[1]/2)]
        lon_m = longitude[int(longitude.shape[0]/2),int(longitude.shape[1]/2)]

        min_lon = np.min(longitude_m)
        max_lon = np.max(longitude_m)
        min_lat = np.min(latitude_m)
        max_lat = np.max(latitude_m)

        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat


        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        data_ms = datam[:,:,4] #pick band (5th band shows ship tracks)

        longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)

        #Define swath and project to raw lat/lon 
        swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
        cellSize = 0.01
        x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
        nx = int(np.floor((max_lon - min_lon) / cellSize))
        ny = int(np.floor((max_lat - min_lat) / cellSize))
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        lon_g, lat_g = np.meshgrid(x, y)
        grid_def = GridDefinition(lons=lon_g, lats=lat_g)
        np.shape(grid_def)
        ri = 10000 #play around with this to fill image, this value seems to work well 
        result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon

        #IMPUTE NANs with minimum value across non-nan pixels 
        result_imp = np.where(np.isnan(result), np.nanmin(result), result) 

        plt.figure(figsize=(20.30*scale_factor,13.54*scale_factor),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,max_lat,min_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, result_imp)

        yearday = labelfiles[i].split('.')[1][1:]
        hourminute = labelfiles[i].split('.')[2]
        plt.savefig(outpath + yearday + '_' + hourminute + '_Data.png',dpi=100, bbox_inches='tight', pad_inches = 0)


        if label:
            #### Now Mask Part
            scaled_label = np.asarray(Image.open(labelfiles[i]))[:,:,1]
            data = np.copy(scaled_label)
            data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
            datam = (data - add_offset) * scale_factor 
            data_m = np.ma.masked_array(datam, np.isnan(datam))
            #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
            mask = data_m.mask
            datam = data_m.data 
            data_ms = datam

            longitude_s = cv.resize(longitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
            latitude_s = cv.resize(latitude_m, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
            #Have to flip shape because cv2 takes in coordinates y,x instead of x,y
            #Define swath and project to raw lat/lon 
            swathDef = SwathDefinition(lons=longitude_s, lats=latitude_s)
            cellSize = 0.01
            x0, xinc, y0, yinc = (min_lon, cellSize, max_lat, -cellSize)
            nx = int(np.floor((max_lon - min_lon) / cellSize))
            ny = int(np.floor((max_lat - min_lat) / cellSize))
            x = np.linspace(x0, x0 + xinc*nx, nx)
            y = np.linspace(y0, y0 + yinc*ny, ny)
            lon_g, lat_g = np.meshgrid(x, y)
            grid_def = GridDefinition(lons=lon_g, lats=lat_g)
            np.shape(grid_def)
            ri = 10000 #play around with this to fill image, this value seems to work well 
            result = resample_nearest(swathDef,data_ms,grid_def,radius_of_influence=ri,epsilon=1,fill_value=np.nan) #can also play around with epsilon
            #IMPUTE NANs with minimum value across non-nan pixels 
            result_imp = np.where(np.isnan(result), np.nanmin(result), result) 

            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 

            ax.set_extent([min_lon,max_lon,max_lat,min_lat], crs=ccrs.PlateCarree(central_longitude=0))
            filled_c = ax.pcolormesh(x, y, result_imp)

            plt.savefig(outpath + yearday + '_' + hourminute + '_Label.png',dpi=100, bbox_inches='tight', pad_inches = 0)




