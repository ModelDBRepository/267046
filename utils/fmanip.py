# -*- coding: utf-8 -*-
"""
Purpose
-------
To conveniently retrieve and manipulate files.

Classes
-------
None

Functions
---------
fetch_files(root_folder, file_name)
    Finds all files with a specific name within the root directory
    and its sub directories.   
sort_files_by_modtime(root_folder, file_name)
    Retrieves all files with a specific name and sorts them by when
    they were last modified (oldest to newest).
merge_csvs(src, dest, stack=[], delim=',')
    Merges two .csv files.
merge_npzs(src, dest, stack=[])
    Merges two .npz files.
merge_fldrs(src, dest, stack=[], omit_files=[], report=True)
    Merge the data in two files unless they are to be omitted.
merge_dir(src, dest, **kwargs)
    Merge two directories. Goes through all of the folders
    in the src and dest root directories and will merge them. Does 
    not merge files in the root.


Change log
----------     
10 Feb 21 - Separated these functions from a more general toolkit.

"""

from shutil import move, rmtree
from os import listdir, rename, walk
from os.path import exists, isdir, getmtime, splitext
import datetime

from numpy import (array, argsort, copy, genfromtxt, hstack,
                   load, savetxt, savez_compressed, zeros)

def fetch_files(root_folder, file_name):
    '''
    Finds all files with a specific name within the root directory
    and its sub directories.
    
    Parameters
    ----------
    root_folder : string
        The root folder that will have its subdirectories searched.
    file_name : string
        The name of the files that need to be found.

    Returns
    -------
    f_list : array(string)
        A list of the files - with their paths - that were found in 
        and above the root drectory.

    '''
    f_list = []

    # Goes through all of the directories and finds all of the files
    #  that match the file_name    
    for root, dirs, files in walk(root_folder):
        if file_name in files:
            f_list.append(root + '/' + file_name)
        
    return f_list

def sort_files_by_modtime(root_folder, file_name):
    '''
    Retrieves all files with a specific name and sorts them by when
    they were last modified (oldest to newest).

    Parameters
    ----------
    root_folder : string
        The root folder that will have its subdirectories searched.
    file_name : string
        The name of the files that need to be found.

    Returns
    -------
    f_list : np.array(string)
        The list of retrieved files in the order of when they were
        last modified.
    datetimes : list(string)
        Time stamps of each files last modification time.

    '''
    f_list = fetch_files(root_folder, file_name)
    
    times = []
    for f in f_list:
        times.append(getmtime(f))
    sorting = argsort(times)
    times = array(times)[sorting]
    f_list = array(f_list)[sorting]
    
    # make the times information easier to read
    datetimes = []
    for t in times:
        datetimes.append(datetime.datetime.fromtimestamp(t))
        
    return f_list, datetimes

def merge_csvs(src, dest, stack=[], delim=','):
    '''
    Merges two .csv files.

    Parameters
    ----------
    src : string
        The file to merge with the dest.
    dest : string
        The file to be merged with.
    stack : list(string)
        The columns that should be stacked (i.e,. adding the last value
        in the dest file to the values in the src file). Default is [].
    delim : string, optional
        The text file's delimiter. The default is ','.

    Returns
    -------
    None.

    '''
    # Loads both files as string data types because it is the
    #   safest format
    src_dat = genfromtxt(src, delimiter=delim, 
                           names=True, dtype=None, 
                           encoding='ascii'
                           )
    dest_dat = genfromtxt(dest, delimiter=delim, 
                            names=True, dtype=None, 
                            encoding='ascii'
                            )
    
    # Edits the source file so that it has the proper trial IDs
    for col in stack:
        if (col in dest_dat.dtype.names) and (col in src_dat.dtype.names):
            src_dat[col] += dest_dat[col][-1]
        else:
            raise ValueError('The column', col, 'does not exist in',
                             'one of the files.')
    
    # Combines the data 
    full_dat = hstack((dest_dat, src_dat))
    
    # Writes the data
    with open(dest, 'w') as fp:
        fp.write('# ' + ','.join(full_dat.dtype.names) + '\n')
        savetxt(fp,full_dat, '%s', ',')

def merge_npzs(src, dest, stack=[]):
    '''
    Merges two .npz files.

    Parameters
    ----------
    src : string
        The file to merge with the dest.
    dest : string
        The file to be merged with.
    stack : list(string)
        The columns that should be stacked (i.e,. adding the last value
        in the dest file to the values in the src file). Default is [].

    Returns
    -------
    None.

    '''
    # Load data
    src_dat = load(src)
    dest_dat = load(dest)
    
    empty = zeros(dest_dat.f.trial.size + src_dat.f.trial.size)
    new_dat = {}
    
    size = dest_dat.f.trial.size
    for key, values in dest_dat.items():
        new_dat.update({key : copy(empty)})
        new_dat[key][:size] = values
    for key, values in src_dat.items():
        if key in stack:
            values += new_dat[key].max()
        new_dat[key][size:] = values
    
    src_dat.close()
    dest_dat.close()
    savez_compressed(dest ,**new_dat)

def merge_fldrs(src, dest, 
                stack=[], omit_files=[], report=True):
    '''
    Merge the data in two files unless they are to be omitted.

    Parameters
    ----------
    src : string
        The folder to merge with the dest.
    dest : string
        The folder to be merged with.
    stack : list(string)
        The columns that should be stacked (i.e,. adding the last value
        in the dest file to the values in the src file). 
        The default is ['trial'].
    omit_files : list(string), optional
        Files that should be omitted from the merger. The default is [].
    report : bool, optional
        Whether the results of the merging should be reported. 
        The default is True.

    Returns
    -------
    None.

    '''
    # Go through each of the files
    for file in listdir(src):      
        # Check if the file should be merged
        if file not in omit_files:  
            name, pofix = splitext(file)
            
            # Merge the files
            if (pofix == '.txt') or (pofix == '.csv'):
                merge_csvs(src+file, dest+file, stack=stack)
            elif pofix == '.npz':
                merge_npzs(src+file, dest+file, stack=stack)
                
            if report: print("Merged", src+file, "with", dest+file)
        else:
            # Simply moves the file
            if not exists(dest + file):                            
                rename(src + file, dest + file)            
                if report: print("Moved", src + file, "to", dest + file)            
            else:
                # Creates a new name for the src file by adding a digit
                #    before the postfix
                counter = 2
                while True:
                    new_name = dest + file.replace('.', str(counter)+'.')
                    if not exists(new_name):
                        rename(src + file, new_name)            
                        if report: print("Renamed", src + file, "to", new_name)
                        break
                    else:
                        counter += 1
            
    # Delete the old folder and its contents
    rmtree(src)

def merge_dir(src, dest, **kwargs):
    '''
    Merge two directories. Will go through all of the folders
    in the two root directories and will merge them. Does not
    merge files in the root.

    Parameters
    ----------
    src : string
        The root folder to merge with the dest.
    dest : string
        The root folder to be merged with.
    kwargs <- may add kwargs for the function merge_fldrs.

    Returns
    -------
    None.

    '''
    # Gets the list of folders for both directories
    dest_fldrs = []
    src_fldrs = []
    for unknown in listdir(dest): 
        if isdir(dest + unknown + '/'):
            dest_fldrs.append(unknown + '/')
    for unknown in listdir(src): 
        if isdir(src + unknown + '/'): 
            src_fldrs.append(unknown + '/') 
    
    # Go through the second directory and move the files to the first directory
    for f in src_fldrs:
        # Ensure that the same folder exists in dir 1
        if f in dest_fldrs:                   
            merge_fldrs(src+f, dest+f, **kwargs)
        else:
            # If no identical folder, then move the folder to the new location
            move(src+f, dest+f)