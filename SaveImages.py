from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import cartopy.crs as ccrs
from PIL import Image
import skimage
from pyresample.geometry import GridDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from pyresample.image import ImageContainerNearest


def SaveRawImages(datafiles,outpath,image_height,image_width):
    
    '''
    Arguments:
    datafiles --> A list of the full paths to raw hdf MODIS files you wish to save as png's (list)
    outpath --> String to folder you wish to save all images too (string)
    image_height --> A float of the desired image height for the final png (float)
    image_width --> A float of the desired image width for the final png (float)

    Outputs
    Year_HourMinute_Raw.png --> A png of the raw image saved in outpath folder (png)
    '''

    for i in range(len(datafiles)):
        # Get the image
        hdf = SD(datafiles[i], SDC.READ)
        image = hdf.select('Atm_Corr_Refl')[:,:,4]
        # Get the date of the observation
        yearday = datafiles[i].split('.')[1][1:]
        hourminute = datafiles[i].split('.')[2]
        plt.figure(figsize=(image_height,image_width),facecolor = 'None')
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(outpath+ yearday + '_' + hourminute + '_Raw.png', dpi = 100, bbox_inches='tight', pad_inches = 0)
        plt.close()

def SaveProjection(datafiles, labelfiles, outpath,image_height,image_width):

    '''
    Arguments:
    datafiles --> A list of the full paths to raw hdf MODIS files you wish to save as projected png's (list)
    labelfiles --> A list of the full paths to labelled png's you wish to save as projected labelled png's (list)
    outpath --> String to folder you wish to save all images too (string)
    image_height --> A float of the desired image height for the final png (float)
    image_width --> A float of the desired image width for the final png (float)

    Outputs
    Year_HourMinute_Raw.png --> A png of the projected image saved in outpath folder (png)
    Year_HourMinute_Masked.png --> A png of the projected masked image saved in outpath folder (png)
    '''

    for i in range(0,len(datafiles)):
        ### Project and Save the Raw Data
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
        # Reposition data that crosses the Greenwich Meridian
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
        data[data == _FillValue] = np.min(data) #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        # Find middle location for plotting
        min_lon = np.min(longitude)
        max_lon = np.max(longitude)
        min_lat = np.min(latitude)
        max_lat = np.max(latitude)
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
        plt.figure(figsize=(image_height,image_width),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, np.flip(data_m2*255.,axis=0))
        # Save Figure
        yearday = datafiles[i].split('.')[1][1:]
        hourminute = datafiles[i].split('.')[2]
        plt.savefig(outpath + yearday + '_' + hourminute + '_Raw.png',dpi=100, bbox_inches='tight', pad_inches = 0)

        ### Project and Save the Masked Label
        label = np.asarray(Image.open(labelfiles[i]))[:,:,0:3]
        scale_factor = 2030 / label.shape[0]
        temp = skimage.transform.rescale(label, scale_factor)
        temp = temp[:,:,4]
        scaled_label = np.zeros([y.shape[0],x.shape[0]])
        if (temp.shape[0] <= y.shape[0]) and (temp.shape[1] <= x.shape[0]):
            scaled_label[:temp.shape[0], :temp.shape[1]] = temp[:,:]
        elif (temp.shape[0] >= y.shape[0]) and (temp.shape[1] >= x.shape[0]):
            scaled_label = temp[:y.shape[0], :x.shape[0]]

        # PLOTTING Labels
        plt.figure(figsize=(image_height,image_width),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,min_lat,max_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, np.flip(scaled_label*255.,axis=0))
        # Save Figure
        plt.savefig(outpath + yearday + '_' + hourminute + '_Masked.png', dpi = 100, bbox_inches='tight', pad_inches = 0)    


def SaveSwath(datafiles, labelfiles, outpath,image_height,image_width):

    '''
    Arguments:
    datafiles --> A list of the full paths to raw hdf MODIS files you wish to save as projected swath png's (list)
    labelfiles --> A list of the full paths to labelled png's you wish to save as projected swath labelled png's (list)
    outpath --> String to folder you wish to save all images too (string)
    image_height --> A float of the desired image height for the final png (float)
    image_width --> A float of the desired image width for the final png (float)

    Outputs
    Year_HourMinute_Raw.png --> A png of the projected swath image saved in outpath folder (png)
    Year_HourMinute_Masked.png --> A png of the projected swath masked image saved in outpath folder (png)
    '''

    for i in range(0,len(datafiles)):
        # Project Raw Data onto a Swath and save the figure
        file = labelfiles[i]
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
        # Reposition data that crosses the Greenwich Meridian
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
        data[data == _FillValue] = np.nan #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))

        # Find min/max Lat and Long 
        min_lon = np.min(longitude)
        max_lon = np.max(longitude)
        min_lat = np.min(latitude)
        max_lat = np.max(latitude)

        x = np.linspace(min_lon, max_lon, np.shape(data_m)[1]) 
        y = np.linspace(min_lat, max_lat, np.shape(data_m)[0]) #lat

        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        mask = data_m.mask
        datam = data_m.data 
        data_ms = datam[:,:,4] #pick band (5th band shows ship tracks best)

        longitude_s = cv.resize(longitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
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

        ## Now Plot Swath
        plt.figure(figsize=(image_height,image_width),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,max_lat,min_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, result_imp)
        # Find Date for saved filename
        yearday = labelfiles[i].split('.')[1][1:]
        hourminute = labelfiles[i].split('.')[2]
        plt.savefig(outpath + yearday + '_' + hourminute + '_Raw.png',dpi=100, bbox_inches='tight', pad_inches = 0)

        #### Now Project Labelled Image onto Swath and save the figure
        scaled_label = np.asarray(Image.open(labelfiles[i]))[:,:,1]
        data = np.copy(scaled_label)
        data[data == _FillValue] = 0 #Data quality flag - change if you don't want nans 
        datam = (data - add_offset) * scale_factor 
        data_m = np.ma.masked_array(datam, np.isnan(datam))
        #PICK BAND AND RESIZE LAT/LON TO SHAPE OF DATA
        data_ms = data_m.data 
        longitude_s = cv.resize(longitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
        latitude_s = cv.resize(latitude, dsize=np.flip(np.shape(data_ms)), interpolation=cv.INTER_CUBIC)
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

        ## Plot Labelled Swath
        plt.figure(figsize=(image_height,image_width),facecolor = 'None')
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0)) 
        ax.set_extent([min_lon,max_lon,max_lat,min_lat], crs=ccrs.PlateCarree(central_longitude=0))
        filled_c = ax.pcolormesh(x, y, result_imp)
        plt.savefig(outpath + yearday + '_' + hourminute + '_Masked.png',dpi=100, bbox_inches='tight', pad_inches = 0)