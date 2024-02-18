'''data.py
Reads CSV files, stores data, access/filter data by variable name
Maddie Puzon
CS 251/2: Data Analysis and Visualization
Spring 2024
'''

import numpy as np


class Data:
    '''Represents data read in from .csv files
    '''
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, cats2levels=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        cats2levels: Python dictionary or None.
                Maps each categorical variable header (var str name) to a list of the unique levels (strings)
                Example:

                For a CSV file that looks like:

                letter,number,greeting
                categorical,categorical,categorical
                a,1,hi
                b,2,hi
                c,2,hi

                cats2levels looks like (key -> value):
                'letter' -> ['a', 'b', 'c']
                'number' -> ['1', '2']
                'greeting' -> ['hi']
        '''
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        self.cats2levels = cats2levels
        
        if self.filepath != None:
            self.read(self.filepath)

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called `self.data` at the end
        (think of this as a 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if there should be nothing returned

        TODO:
        1. Set or update your `filepath` instance variable based on the parameter value.
        2. Open and read in the .csv file `filepath` to set `self.data`.
        Parse the file to ONLY store numeric and categorical columns of data in a 2D tabular format (ignore all other
        potential variable types).
            - Numeric data: Store all values as floats.
            - Categorical data: Store values as ints in your list of lists (self.data). Maintain the mapping between the
            int-based and string-based coding of categorical levels in the self.cats2levels dictionary.
        All numeric and categorical values should be added to the SAME list of lists (self.data).
        3. Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        4. Be sure to set the fields: `self.headers`, `self.data`, `self.header2col`, `self.cats2levels`.
        5. Add support for missing data. This arises with there is no entry in a CSV file between adjacent commas.
            For example:
                    letter,number,greeting
                    categorical,categorical,categorical
                     a,1,hi
                     b,,hi
                     c,,hi
            contains two missing values, in the 4th and 5th rows of the 2nd column.
            Handle this differently depending on whether the missing value belongs to a numeric or categorical variable.
            In both cases, you should subsitute a single constant value for the current value to your list of lists (self.data):
            - Numeric data: Subsitute np.nan for the missing value.
            (nan stands for "not a number" — this is a special constant value provided by Numpy).
            - Categorical data: Add a categorical level called 'Missing' to the list of levels in self.cats2levels
            associated with the current categorical variable that has the missing value. Now proceed as if the level
            'Missing' actually appeared in the CSV file and make the current entry in your data list of lists (self.data)
            the INT representing the index (position) of 'Missing' in the level list.
            For example, in the above CSV file example, self.data should look like:
                [[0, 0, 0],
                 [1, 1, 0],
                 [2, 1, 0]]
            and self.cats2levels would look like:
                self.cats2levels['letter'] -> ['a', 'b', 'c']
                self.cats2levels['number'] -> ['1', 'Missing']
                self.cats2levels['greeting'] -> ['hi']

        NOTE:
        - In any CS251 project, you are welcome to create as many helper methods as you'd like. The crucial thing is to
        make sure that the provided method signatures work as advertised.
        - You should only use the basic Python to do your parsing. (i.e. no Numpy or other imports).
        Points will be taken off otherwise.
        - Have one of the CSV files provided on the project website open in a text editor as you code and debug.
        - Run the provided test scripts regularly to see desired outputs and to check your code.
        - It might be helpful to implement support for only numeric data first, test it, then add support for categorical
        variable types afterward.
        - Make use of code from Lab 1a!
        '''
        
        self.filepath = filepath
        
        with open(self.filepath) as f:
            
            data = []
            # get all lines from .csv file
            for line in f:
                lines = line.split(',')
                stripped_lines = []
                for element in lines:
                    stripped_lines.append(element.strip())
                data.append(stripped_lines)
            # check if file format is correct--1st row headers, 2nd row header types
            valid_header_types = ['string', 'numeric', 'categorical', 'date']
            for element in data[1]:
                if element not in valid_header_types:
                    print("ERROR: invalid header categories. Make sure first and second lines of csv file are headers and categories respectively.")
                    print("Invalid element: ", "\"", element, "\"")
                    exit()
            # assign headers
            self.headers = data[0]
            self.header_types = data[1]

        # determine which cols to copy to self.data
        data_types = ['numeric', 'categorical']
        cols_to_keep = []
        for i in range(len(data[1])):
            if data[1][i] in data_types:
                cols_to_keep.append(i)
        # loop through data list and append proper columns to self.data
        self.data = []
        for sample in data[2:]:
            num_cat_sample = []
            for element in sample:
                if sample.index(element) in cols_to_keep:
                    # turn into float if possible
                    if element.isdigit():
                        num_cat_sample.append(float(element))
                    else:   # append string element
                        num_cat_sample.append(element)
            self.data.append(num_cat_sample)
        # update headers and header_types lists to match self.data
        new_headers = []
        new_header_types =[]
        for header in self.headers:
            if self.headers.index(header) in cols_to_keep:
                new_headers.append(header)
                new_header_types.append(self.header_types[self.headers.index(header)])
        self.headers = new_headers
        self.header_types = new_header_types
        
        # create dictionary to map headers (string) to column index (int)
        h2i = {}
        for i in range(len(self.headers)):
            h2i[self.headers[i]] = i
        self.header2col = h2i

        # set self.cats2levels
        self.cats2levels = {}
        for i in range(len(self.headers)):
            if self.header_types[i] == 'categorical':
                self.cats2levels[self.headers[i]] = []
                for sample in self.data:
                    #  print(i) # 14 causing problems
                    if sample[i] not in self.cats2levels[self.headers[i]]:
                        self.cats2levels[self.headers[i]].append(sample[i])

        for key in self.cats2levels:
            for element in self.cats2levels[key]:
                if element == '':
                    self.cats2levels[key][self.cats2levels[key].index(element)] = 'Missing'

        # set self.header2col
        self.header2col = {}
        for i in range(len(self.headers)):
            self.header2col[self.headers[i]] = i
            
        # handle missing data
        
        # map categorical variable string to int from cats2levels
        to_replace = [] # contains indicies of element to replace in each sample
        for i in range(len(self.headers)):
            if self.header_types[i] == 'categorical': # get index to replace in each sample
                to_replace.append(i)
        for sample in self.data:
            print(f'SAMPLE: {sample}')
            for element in sample:
                if sample.index(element) in to_replace: # replace string value with corresponding int from cats2levels
                    if element == '':
                        sample[sample.index(element)] = 'Missing'
                    else:
                        print(f'element: {element}')
                        print(f'looking in: {self.cats2levels[self.headers[sample.index(element)]]}')
                        # get levels back to categories for comparison
                        sample[sample.index(element)] = float(self.cats2levels[self.headers[sample.index(element)]].index(element))
                else: # numeric data
                    if element == '':
                        sample[sample.index(element)] = np.nan
                # # for numeric rows
                # else: # turn into float or 'np.nan' if missing
                #     if element == '': # if missing
                #         sample[sample.index(element)] = np.nan
                #     else: # turn into float
                #         sample[sample.index(element)] = float(element)

        self.data = np.array(self.data)
         

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
        what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.

        NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
        Printing out the categorical variables with string levels would be a small extension.
        '''
        pass

    def get_headers(self):
        '''Get list of header names (all variables)

        Returns:
        -----------
        Python list of str.
        '''
        pass

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        pass

    def get_cat_level_mappings(self):
        '''Get method for mapping between categorical variable names and a list of the respective unique level strings.

        Returns:
        -----------
        Python dictionary. str -> list of str
        '''
        pass

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        pass

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        pass

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        pass

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers` list.
        '''
        pass

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself. This can be accomplished with numpy's copy
            function.
        '''
        pass

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        pass

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        pass

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        pass

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return column #2 of self.data.
        If rows is not [] (say =[0, 2, 5]), then we do the same thing, but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select. Empty list [] means take all rows.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        pass
