import pandas as pd


class BatchIterator ():
        """
        split an iterable into mini batch with batch length of batch_number
        supports batch of a pandas dataframe
        usage:
        
            for i in batch([1,2,3,4,5], batch_number=2):
                print(i)
            
            for idx, mini_data in enumerate(batch(df, batch_number=10)):
                print(idx)
                print(mini_data)
                
        @link https://stackoverflow.com/questions/25699439/how-to-iterate-over-consecutive-chunks-of-pandas-dataframe-efficiently
        """


    def batch (iterable, batch_number = 100):
        l = len (iterable)

        for idx in range (0, l, batch_number):
            if isinstance (iterable, pd.DataFrame):
                # dataframe can't split index label, should iter according index
                yield iterable.iloc[idx:min(idx+batch_number, l)]
            else:
                yield iterable[idx:min(idx+batch_number, l)]