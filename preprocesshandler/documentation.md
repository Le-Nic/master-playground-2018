**config** file in `/configs` are evaluated using `eval` and will be stored as dictionary, allowing any Python compatible code to be executed.

- io_csv, *Dictionary*  
Uses CSV as input

    - train_dir, *String*  
    Directory / File path of train set.
    
    - test_dir, *String / None*  
    Directory / File path of test set.
    
    - output_dir, *String*  
    Directory where the processed file will be stored (directory will be created if not exist).
    
    - output_type, *String: (hd5 / csv)*  
    The output type of the processed file. Data type for hd5 will be Float64Atom, and csv *%.16f*.
    
    - read_chunk_size, *Integer*  
    Number of chunks/lines to be read in a single iteration.
    
    - delimiter, *String*  
    The delimiter used by the data set when in csv format.
    
    - header, *Boolean: (True / None)*  
    Treating first row as header / data.
    
    - dates, *List [Integer]*  
    Columns in which Pandas csv reader will parse it into *datetime* object.  
    **All columns specified must be omitted in `io_csv.dtypes_in`.**  
    
        **To be safe, enter only 1 column.**
    
    - is_epoch, *Boolean*  
    Treating all columns in `pp.t` as epoch instead of *datetime*.  
    Parser will assume epoch used is in *%s.%µs* format.
    
    - dtypes_in, *Dictionary {Integer : np.dtype}*  
    Data type for each column. Includes all columns for label(s), they are stored in a separate file.
        > { i : np.dtype }
    
- normalization, *String: (minmax1r / minmax2r / zscore)*  
Apply selected normalization technique to all columns specified in `pp.norm` and `pp.t`.
    1. minmax1r  
    Scale data into range of 0 - 1.
        > (x<sub>i</sub> - min<sub>i</sub>) / (max<sub>i</sub> - min<sub>i</sub>)
    2. minmax2r
    Scale data into range of -1 - 1.
        > 2 * (x<sub>i</sub> - min<sub>i</sub>) / (max<sub>i</sub> - min<sub>i</sub>) - 1
    3. zscore
    Normalize data to have 0 mean and unit variance.
        > x - μ<sub>i</sub>) / σ<sub>i</sub>
    
        **Note: Z-score does not work for `pp.t` data (not implemented)**

- add_td, *Integer / None*  
Values in this column will be added into all columns in `pp.t`.  
Before that, the values will be scaled based on `td_scale` to treat it as *%s.%µs* format.
    > column<sub>add_td</sub> * td_scale + column<sub>pp.t</sub> 

    Required to convert *ts* (time start / first seen) into *te* date (time end / last seen).

- td_scale, *Float*  
Value in which `add_td` will be multiplied to prior adding them with *ts*,
converting the values in `add_td` column into seconds.

- label, *Dictionary*
    - i, *List [Integer]*  
    Columns to be treated as label(s), first column before the specified will be treated as data.  
    If more than 1 column is specified, labels in differing columns will be used in creation of mixed labels.  
    
        **To be safe, enter only 1-2 column(s)**
        ```
        E.g. output of [[a, b, c], [d, e, f, g]], assuming a and d is benign label:
        
        class-0:
            a -> 0
            b -> 1
            c -> 1
        
        class-1:
            a -> 0
            b -> 1
            c -> 2
        
        class-2:
            [a, d] -> 0
            e -> 1
            f -> 2
            g -> 3
        
        class-3:
            [a, d] -> 0
            [b, e] -> 1
            [b, f] -> 2
            [b, g] -> 3
            [c, e] -> 4
            [c, f] -> 5
            [c, g] -> 6
        ```
    
    - lbl_normal, *List []*  
    Value of the label which is treated as being benign / normal.  
    They are used in mapping of the labels.
      
        **Values specified must have same size and alignment/index with the columns specified in `label.i`.**

- pp, *Dictionary*
    - t, *List [Integer]* **(only works in: convert_hd5)**  
    Perform day & week cyclical transformation, resulting in extra 3 columns each.  
    If `add_td` column is specified, The value of the specified column will be added after `td_scale` is multipled with.
    
    - ips, *Lists [Integer]* **(only works in: convert_hd5)**  
    Treating specified column(s) as IP, saving one hot encoded index in seperate HD5 node (`output_type`: hd5), 
    with the real IP value saved in meta file.
    
        **To be safe, enter exactly 2 columns**
    
    - int, *Lists [Integer]* **(only works in: convert_hd5)**  
    Numericalize column which has *str / obj* data type. Imagine one-hot encoding without expanding dimensions.
    
    - 1hot, *Lists [Integer]*  
    Perform one hot encoding.
    
    - flg, *Lists [Integer]*  
    Convert flags into binary form.  
    Allow only column(s) which have exactly 6 characters as value. For each character,
    dot (.) will be treated as 0, and others 1.
    
    - 8bit, *Lists [Integer]*  
    Convert integers (0-255) into 8-bit binary form.
    
    - 16bit, *Lists [Integer]*  
    Treating specified column(s) as Ports, converting the values into 16-bit binary form.  
    Columns specified here should uses *np.object* in `io_csv.dtypes_in`.  
    If decimals are detected, it will be treated as ICMP and are stored as *[8-bit ICMP code, 8-bit ICMP type]*.

    - norm, *Lists [Integer]*  
    Perform normalization technique specified in `normalization`.
    
    - rm, *Lists [Integer]*
    Column(s) specified here will be removed.
