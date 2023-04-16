# Recommendation-Engine
A recommendation ssytem suggest products/items to users based on certain criteria.

In this project, I build a recommendation engine for implicit data (data nased on user behaviour iwth no ratings or actions) using Alternating Least Squares (ALS).


Import Libraries

#import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

Import Dataset from SQL Server

#importing your dataset from sql
db = 'AdventureWorksDW2019' #databaseName
server = 'DESKTOP-RDRRI0S'  #serverName

#create the connection
conn = pyodbc.connect('DRIVER={SQL SERVER};SERVER=' + server + ';DATABASE=' + db + ';Trusted_Connection=yes')



#query 
query = """
SELECT FS.ProductKey,
	   FS.CustomerKey,
	   FS.OrderQuantity,
	   FS.OrderDate,
	   DP.EnglishProductName AS Product_Description,
	   DSP.ProductSubcategoryKey,
	   DSP.EnglishProductSubcategoryName,
	   DCP.EnglishProductCategoryName
FROM FactInternetSales AS FS
LEFT JOIN DimProduct AS DP
ON FS.ProductKey = DP.ProductKey
LEFT JOIN DimProductSubCategory AS DSP
ON DP.ProductSubCategoryKey = DSP.ProductSubCategoryKey
LEFT JOIN DimProductCategory AS DCP
ON DSP.ProductCategoryKey = DCP.ProductCategoryKey
"""

#reading the sql file from the engine
dataset = pd.read_sql(query, conn) 

dataset.head()

	ProductKey 	CustomerKey 	OrderQuantity 	OrderDate 	Product_Description 	ProductSubcategoryKey 	EnglishProductSubcategoryName 	EnglishProductCategoryName
0 	310 	21768 	1 	2010-12-29 	Road-150 Red, 62 	2 	Road Bikes 	Bikes
1 	346 	28389 	1 	2010-12-29 	Mountain-100 Silver, 44 	1 	Mountain Bikes 	Bikes
2 	346 	25863 	1 	2010-12-29 	Mountain-100 Silver, 44 	1 	Mountain Bikes 	Bikes
3 	336 	14501 	1 	2010-12-29 	Road-650 Black, 62 	2 	Road Bikes 	Bikes
4 	346 	11003 	1 	2010-12-29 	Mountain-100 Silver, 44 	1 	Mountain Bikes 	Bikes

#check to see if there are any missing values

dataset.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60398 entries, 0 to 60397
Data columns (total 8 columns):
 #   Column                         Non-Null Count  Dtype         
---  ------                         --------------  -----         
 0   ProductKey                     60398 non-null  int64         
 1   CustomerKey                    60398 non-null  int64         
 2   OrderQuantity                  60398 non-null  int64         
 3   OrderDate                      60398 non-null  datetime64[ns]
 4   Product_Description            60398 non-null  object        
 5   ProductSubcategoryKey          60398 non-null  int64         
 6   EnglishProductSubcategoryName  60398 non-null  object        
 7   EnglishProductCategoryName     60398 non-null  object        
dtypes: datetime64[ns](1), int64(4), object(3)
memory usage: 3.7+ MB

#returns only unique product/name pairs
item_lookup = dataset[['ProductKey', 'Product_Description']].drop_duplicates() 

#Encode ProductKey as string for easy future lookup
item_lookup['ProductKey'] = item_lookup.ProductKey.astype(str) 

item_lookup.head()

	ProductKey 	Product_Description
0 	310 	Road-150 Red, 62
1 	346 	Mountain-100 Silver, 44
3 	336 	Road-650 Black, 62
5 	311 	Road-150 Red, 44
7 	351 	Mountain-100 Black, 48

dataset['ProductKey'] = dataset.ProductKey.astype(str) #convert ProductKey to str data type

dataset.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60398 entries, 0 to 60397
Data columns (total 8 columns):
 #   Column                         Non-Null Count  Dtype         
---  ------                         --------------  -----         
 0   ProductKey                     60398 non-null  object        
 1   CustomerKey                    60398 non-null  int64         
 2   OrderQuantity                  60398 non-null  int64         
 3   OrderDate                      60398 non-null  datetime64[ns]
 4   Product_Description            60398 non-null  object        
 5   ProductSubcategoryKey          60398 non-null  int64         
 6   EnglishProductSubcategoryName  60398 non-null  object        
 7   EnglishProductCategoryName     60398 non-null  object        
dtypes: datetime64[ns](1), int64(3), object(4)
memory usage: 3.7+ MB

#get rid of unnecessary info, keep only ProductKey, CustomerKey and Quantity
dataset = dataset[['CustomerKey', 'ProductKey', 'OrderQuantity']]
dataset

	CustomerKey 	ProductKey 	OrderQuantity
0 	21768 	310 	1
1 	28389 	346 	1
2 	25863 	346 	1
3 	14501 	336 	1
4 	11003 	346 	1
... 	... 	... 	...
60393 	15868 	485 	1
60394 	15868 	225 	1
60395 	18759 	485 	1
60396 	18759 	486 	1
60397 	18759 	225 	1

60398 rows × 3 columns

#Group data together
grouped_dataset = dataset.groupby(['CustomerKey', 'ProductKey']).sum().reset_index() 

grouped_dataset

	CustomerKey 	ProductKey 	OrderQuantity
0 	11000 	214 	1
1 	11000 	344 	1
2 	11000 	353 	1
3 	11000 	485 	1
4 	11000 	488 	1
... 	... 	... 	...
59046 	29480 	479 	1
59047 	29480 	562 	1
59048 	29481 	349 	1
59049 	29482 	358 	1
59050 	29483 	360 	1

59051 rows × 3 columns

#Replace a sum of zero purchases with a one to indicate purchased
grouped_dataset.OrderQuantity.loc[grouped_dataset.OrderQuantity == 0] = 1 

C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexing.py:670: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  iloc._setitem_with_indexer(indexer, value)

#Get customers where purchase totals were positive
grouped_purchased = grouped_dataset.query('OrderQuantity > 0') 

grouped_purchased.head()

	CustomerKey 	ProductKey 	OrderQuantity
0 	11000 	214 	1
1 	11000 	344 	1
2 	11000 	353 	1
3 	11000 	485 	1
4 	11000 	488 	1

grouped_purchased.info()

<class 'pandas.core.frame.DataFrame'>
Int64Index: 59051 entries, 0 to 59050
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   CustomerKey    59051 non-null  int64 
 1   ProductKey     59051 non-null  object
 2   OrderQuantity  59051 non-null  int64 
dtypes: int64(2), object(1)
memory usage: 1.8+ MB

from pandas.api.types import CategoricalDtype

#Get all unique customers
customers = list(np.sort(grouped_purchased.CustomerKey.unique()))

customers

[11000,
 11001,
 11002,
 11003,
 11004,
 11005,
 11006,
 11007,
 11008,
 11009,
 11010,
 11011,
 11012,
 11013,
 11014,
 11015,
 11016,
 11017,
 11018,
 11019,
 11020,
 11021,
 11022,
 11023,
 11024,
 11025,
 11026,
 11027,
 11028,
 11029,
 11030,
 11031,
 11032,
 11033,
 11034,
 11035,
 11036,
 11037,
 11038,
 11039,
 11040,
 11041,
 11042,
 11043,
 11044,
 11045,
 11046,
 11047,
 11048,
 11049,
 11050,
 11051,
 11052,
 11053,
 11054,
 11055,
 11056,
 11057,
 11058,
 11059,
 11060,
 11061,
 11062,
 11063,
 11064,
 11065,
 11066,
 11067,
 11068,
 11069,
 11070,
 11071,
 11072,
 11073,
 11074,
 11075,
 11076,
 11077,
 11078,
 11079,
 11080,
 11081,
 11082,
 11083,
 11084,
 11085,
 11086,
 11087,
 11088,
 11089,
 11090,
 11091,
 11092,
 11093,
 11094,
 11095,
 11096,
 11097,
 11098,
 11099,
 11100,
 11101,
 11102,
 11103,
 11104,
 11105,
 11106,
 11107,
 11108,
 11109,
 11110,
 11111,
 11112,
 11113,
 11114,
 11115,
 11116,
 11117,
 11118,
 11119,
 11120,
 11121,
 11122,
 11123,
 11124,
 11125,
 11126,
 11127,
 11128,
 11129,
 11130,
 11131,
 11132,
 11133,
 11134,
 11135,
 11136,
 11137,
 11138,
 11139,
 11140,
 11141,
 11142,
 11143,
 11144,
 11145,
 11146,
 11147,
 11148,
 11149,
 11150,
 11151,
 11152,
 11153,
 11154,
 11155,
 11156,
 11157,
 11158,
 11159,
 11160,
 11161,
 11162,
 11163,
 11164,
 11165,
 11166,
 11167,
 11168,
 11169,
 11170,
 11171,
 11172,
 11173,
 11174,
 11175,
 11176,
 11177,
 11178,
 11179,
 11180,
 11181,
 11182,
 11183,
 11184,
 11185,
 11186,
 11187,
 11188,
 11189,
 11190,
 11191,
 11192,
 11193,
 11194,
 11195,
 11196,
 11197,
 11198,
 11199,
 11200,
 11201,
 11202,
 11203,
 11204,
 11205,
 11206,
 11207,
 11208,
 11209,
 11210,
 11211,
 11212,
 11213,
 11214,
 11215,
 11216,
 11217,
 11218,
 11219,
 11220,
 11221,
 11222,
 11223,
 11224,
 11225,
 11226,
 11227,
 11228,
 11229,
 11230,
 11231,
 11232,
 11233,
 11234,
 11235,
 11236,
 11237,
 11238,
 11239,
 11240,
 11241,
 11242,
 11243,
 11244,
 11245,
 11246,
 11247,
 11248,
 11249,
 11250,
 11251,
 11252,
 11253,
 11254,
 11255,
 11256,
 11257,
 11258,
 11259,
 11260,
 11261,
 11262,
 11263,
 11264,
 11265,
 11266,
 11267,
 11268,
 11269,
 11270,
 11271,
 11272,
 11273,
 11274,
 11275,
 11276,
 11277,
 11278,
 11279,
 11280,
 11281,
 11282,
 11283,
 11284,
 11285,
 11286,
 11287,
 11288,
 11289,
 11290,
 11291,
 11292,
 11293,
 11294,
 11295,
 11296,
 11297,
 11298,
 11299,
 11300,
 11301,
 11302,
 11303,
 11304,
 11305,
 11306,
 11307,
 11308,
 11309,
 11310,
 11311,
 11312,
 11313,
 11314,
 11315,
 11316,
 11317,
 11318,
 11319,
 11320,
 11321,
 11322,
 11323,
 11324,
 11325,
 11326,
 11327,
 11328,
 11329,
 11330,
 11331,
 11332,
 11333,
 11334,
 11335,
 11336,
 11337,
 11338,
 11339,
 11340,
 11341,
 11342,
 11343,
 11344,
 11345,
 11346,
 11347,
 11348,
 11349,
 11350,
 11351,
 11352,
 11353,
 11354,
 11355,
 11356,
 11357,
 11358,
 11359,
 11360,
 11361,
 11362,
 11363,
 11364,
 11365,
 11366,
 11367,
 11368,
 11369,
 11370,
 11371,
 11372,
 11373,
 11374,
 11375,
 11376,
 11377,
 11378,
 11379,
 11380,
 11381,
 11382,
 11383,
 11384,
 11385,
 11386,
 11387,
 11388,
 11389,
 11390,
 11391,
 11392,
 11393,
 11394,
 11395,
 11396,
 11397,
 11398,
 11399,
 11400,
 11401,
 11402,
 11403,
 11404,
 11405,
 11406,
 11407,
 11408,
 11409,
 11410,
 11411,
 11412,
 11413,
 11414,
 11415,
 11416,
 11417,
 11418,
 11419,
 11420,
 11421,
 11422,
 11423,
 11424,
 11425,
 11426,
 11427,
 11428,
 11429,
 11430,
 11431,
 11432,
 11433,
 11434,
 11435,
 11436,
 11437,
 11438,
 11439,
 11440,
 11441,
 11442,
 11443,
 11444,
 11445,
 11446,
 11447,
 11448,
 11449,
 11450,
 11451,
 11452,
 11453,
 11454,
 11455,
 11456,
 11457,
 11458,
 11459,
 11460,
 11461,
 11462,
 11463,
 11464,
 11465,
 11466,
 11467,
 11468,
 11469,
 11470,
 11471,
 11472,
 11473,
 11474,
 11475,
 11476,
 11477,
 11478,
 11479,
 11480,
 11481,
 11482,
 11483,
 11484,
 11485,
 11486,
 11487,
 11488,
 11489,
 11490,
 11491,
 11492,
 11493,
 11494,
 11495,
 11496,
 11497,
 11498,
 11499,
 11500,
 11501,
 11502,
 11503,
 11504,
 11505,
 11506,
 11507,
 11508,
 11509,
 11510,
 11511,
 11512,
 11513,
 11514,
 11515,
 11516,
 11517,
 11518,
 11519,
 11520,
 11521,
 11522,
 11523,
 11524,
 11525,
 11526,
 11527,
 11528,
 11529,
 11530,
 11531,
 11532,
 11533,
 11534,
 11535,
 11536,
 11537,
 11538,
 11539,
 11540,
 11541,
 11542,
 11543,
 11544,
 11545,
 11546,
 11547,
 11548,
 11549,
 11550,
 11551,
 11552,
 11553,
 11554,
 11555,
 11556,
 11557,
 11558,
 11559,
 11560,
 11561,
 11562,
 11563,
 11564,
 11565,
 11566,
 11567,
 11568,
 11569,
 11570,
 11571,
 11572,
 11573,
 11574,
 11575,
 11576,
 11577,
 11578,
 11579,
 11580,
 11581,
 11582,
 11583,
 11584,
 11585,
 11586,
 11587,
 11588,
 11589,
 11590,
 11591,
 11592,
 11593,
 11594,
 11595,
 11596,
 11597,
 11598,
 11599,
 11600,
 11601,
 11602,
 11603,
 11604,
 11605,
 11606,
 11607,
 11608,
 11609,
 11610,
 11611,
 11612,
 11613,
 11614,
 11615,
 11616,
 11617,
 11618,
 11619,
 11620,
 11621,
 11622,
 11623,
 11624,
 11625,
 11626,
 11627,
 11628,
 11629,
 11630,
 11631,
 11632,
 11633,
 11634,
 11635,
 11636,
 11637,
 11638,
 11639,
 11640,
 11641,
 11642,
 11643,
 11644,
 11645,
 11646,
 11647,
 11648,
 11649,
 11650,
 11651,
 11652,
 11653,
 11654,
 11655,
 11656,
 11657,
 11658,
 11659,
 11660,
 11661,
 11662,
 11663,
 11664,
 11665,
 11666,
 11667,
 11668,
 11669,
 11670,
 11671,
 11672,
 11673,
 11674,
 11675,
 11676,
 11677,
 11678,
 11679,
 11680,
 11681,
 11682,
 11683,
 11684,
 11685,
 11686,
 11687,
 11688,
 11689,
 11690,
 11691,
 11692,
 11693,
 11694,
 11695,
 11696,
 11697,
 11698,
 11699,
 11700,
 11701,
 11702,
 11703,
 11704,
 11705,
 11706,
 11707,
 11708,
 11709,
 11710,
 11711,
 11712,
 11713,
 11714,
 11715,
 11716,
 11717,
 11718,
 11719,
 11720,
 11721,
 11722,
 11723,
 11724,
 11725,
 11726,
 11727,
 11728,
 11729,
 11730,
 11731,
 11732,
 11733,
 11734,
 11735,
 11736,
 11737,
 11738,
 11739,
 11740,
 11741,
 11742,
 11743,
 11744,
 11745,
 11746,
 11747,
 11748,
 11749,
 11750,
 11751,
 11752,
 11753,
 11754,
 11755,
 11756,
 11757,
 11758,
 11759,
 11760,
 11761,
 11762,
 11763,
 11764,
 11765,
 11766,
 11767,
 11768,
 11769,
 11770,
 11771,
 11772,
 11773,
 11774,
 11775,
 11776,
 11777,
 11778,
 11779,
 11780,
 11781,
 11782,
 11783,
 11784,
 11785,
 11786,
 11787,
 11788,
 11789,
 11790,
 11791,
 11792,
 11793,
 11794,
 11795,
 11796,
 11797,
 11798,
 11799,
 11800,
 11801,
 11802,
 11803,
 11804,
 11805,
 11806,
 11807,
 11808,
 11809,
 11810,
 11811,
 11812,
 11813,
 11814,
 11815,
 11816,
 11817,
 11818,
 11819,
 11820,
 11821,
 11822,
 11823,
 11824,
 11825,
 11826,
 11827,
 11828,
 11829,
 11830,
 11831,
 11832,
 11833,
 11834,
 11835,
 11836,
 11837,
 11838,
 11839,
 11840,
 11841,
 11842,
 11843,
 11844,
 11845,
 11846,
 11847,
 11848,
 11849,
 11850,
 11851,
 11852,
 11853,
 11854,
 11855,
 11856,
 11857,
 11858,
 11859,
 11860,
 11861,
 11862,
 11863,
 11864,
 11865,
 11866,
 11867,
 11868,
 11869,
 11870,
 11871,
 11872,
 11873,
 11874,
 11875,
 11876,
 11877,
 11878,
 11879,
 11880,
 11881,
 11882,
 11883,
 11884,
 11885,
 11886,
 11887,
 11888,
 11889,
 11890,
 11891,
 11892,
 11893,
 11894,
 11895,
 11896,
 11897,
 11898,
 11899,
 11900,
 11901,
 11902,
 11903,
 11904,
 11905,
 11906,
 11907,
 11908,
 11909,
 11910,
 11911,
 11912,
 11913,
 11914,
 11915,
 11916,
 11917,
 11918,
 11919,
 11920,
 11921,
 11922,
 11923,
 11924,
 11925,
 11926,
 11927,
 11928,
 11929,
 11930,
 11931,
 11932,
 11933,
 11934,
 11935,
 11936,
 11937,
 11938,
 11939,
 11940,
 11941,
 11942,
 11943,
 11944,
 11945,
 11946,
 11947,
 11948,
 11949,
 11950,
 11951,
 11952,
 11953,
 11954,
 11955,
 11956,
 11957,
 11958,
 11959,
 11960,
 11961,
 11962,
 11963,
 11964,
 11965,
 11966,
 11967,
 11968,
 11969,
 11970,
 11971,
 11972,
 11973,
 11974,
 11975,
 11976,
 11977,
 11978,
 11979,
 11980,
 11981,
 11982,
 11983,
 11984,
 11985,
 11986,
 11987,
 11988,
 11989,
 11990,
 11991,
 11992,
 11993,
 11994,
 11995,
 11996,
 11997,
 11998,
 11999,
 ...]

#Get all unique products
products = list(grouped_purchased.ProductKey.unique())

products

['214',
 '344',
 '353',
 '485',
 '488',
 '530',
 '541',
 '573',
 '217',
 '225',
 '350',
 '477',
 '478',
 '479',
 '491',
 '604',
 '222',
 '346',
 '359',
 '561',
 '361',
 '480',
 '564',
 '345',
 '355',
 '562',
 '351',
 '528',
 '537',
 '357',
 '347',
 '348',
 '575',
 '489',
 '574',
 '529',
 '465',
 '486',
 '363',
 '569',
 '228',
 '463',
 '467',
 '475',
 '482',
 '483',
 '535',
 '536',
 '538',
 '539',
 '487',
 '490',
 '484',
 '472',
 '565',
 '567',
 '571',
 '349',
 '566',
 '568',
 '570',
 '585',
 '572',
 '481',
 '471',
 '234',
 '473',
 '587',
 '576',
 '231',
 '474',
 '310',
 '540',
 '563',
 '590',
 '313',
 '589',
 '237',
 '312',
 '476',
 '380',
 '578',
 '605',
 '577',
 '584',
 '579',
 '606',
 '360',
 '591',
 '592',
 '378',
 '374',
 '586',
 '580',
 '311',
 '314',
 '356',
 '362',
 '588',
 '354',
 '358',
 '352',
 '583',
 '600',
 '599',
 '597',
 '581',
 '593',
 '376',
 '560',
 '596',
 '598',
 '390',
 '372',
 '594',
 '595',
 '388',
 '382',
 '384',
 '368',
 '375',
 '379',
 '370',
 '369',
 '377',
 '371',
 '373',
 '386',
 '387',
 '385',
 '383',
 '381',
 '582',
 '389',
 '336',
 '328',
 '332',
 '338',
 '330',
 '322',
 '340',
 '326',
 '334',
 '320',
 '342',
 '324',
 '325',
 '321',
 '335',
 '337',
 '333',
 '323',
 '329',
 '341',
 '343',
 '331',
 '327',
 '339']

#Get all purchases
quantity = list(grouped_purchased.OrderQuantity) 

quantity

[1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 5,
 2,
 2,
 2,
 1,
 3,
 3,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 2,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 2,
 3,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 3,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 10,
 1,
 1,
 2,
 3,
 8,
 4,
 4,
 1,
 1,
 2,
 2,
 4,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 3,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 3,
 2,
 2,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 3,
 4,
 2,
 1,
 1,
 1,
 1,
 2,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 2,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 1,
 ...]

#Get the associated row indcies
rows = grouped_purchased.CustomerKey.astype(CategoricalDtype(categories = customers)).cat.codes

#Get the associated column indices
cols = grouped_purchased.ProductKey.astype(CategoricalDtype(categories = products)).cat.codes

#We check our final matrix object
purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))

purchases_sparse

<18484x158 sparse matrix of type '<class 'numpy.intc'>'
	with 59051 stored elements in Compressed Sparse Row format>

#Shows the number of possible interactions in the matrix
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] 

#Shows the number of items interacted with sparsity
num_purchases = len(purchases_sparse.nonzero()[0]) 
sparsity = 100*(1 - (num_purchases/matrix_size))
sparsity

97.97803231806365

97.97% of the interaction matrix is sparse.
For collaborative filtering to work, the maximum sparsity should be about 99.5% or so.
We are well below this, so we should be able to get decent results.

 

Creating a Validation and Testing Set

#Our test set is an exact copy of our original data. 
#The training set, however, will mask a random percentage of user/item interactions and act as if the user never purchased the item (making it a sparse entry with a zero). 
#We then check in the test set which items were recommended to the user that they ended up actually purchasing.
#If the users frequently ended up purchasing the items most recommended to them by the system, we can conclude the system seems to be working.

#As an additional check, we can compare our system to simply recommending the most popular items to every user (beating popularity is a bit difficult). This will be our baseline.

import random

def make_train(ratings, pct_test = 0.2):

    # Make a copy of the original dataset to be the test set
    test_set = ratings.copy()

    # Store the test set as a binary preference matrix
    test_set[test_set != 0] = 1  

    # Make a copy of the original data we can alter as our training set
    training_set = ratings.copy() 

    # Find the indices in the ratings data where an interaction exists
    nonzero_inds = training_set.nonzero() 

    # Zip these pairs together of user,item index into list
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) 

    # Set the random seed to zero for reproducibility
    random.seed(0) 

    # Round the number of samples needed to the nearest integer
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) 

    # Sample a random number of user-item pairs without replacement
    samples = random.sample(nonzero_pairs, num_samples) 

    # Get the user row indices 
    user_inds = [index[0] for index in samples] 

    # Get the item column indices
    item_inds = [index[1] for index in samples] 

    # Assign all of the randomly chosen user-item pairs to zero
    training_set[user_inds, item_inds] = 0 

    # Get rid of zeros in sparse array storage after update to save space
    training_set.eliminate_zeros()  

    # Output the unique list of user rows that were altered
    return training_set, test_set, list(set(user_inds)) 

product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)

##Implementing Alternating Least Square (ALS) algorithm for implicit feedback
def implicit_weighted_ALS(training_set, lambda_val = 0.1, alpha = 40, iterations = 10, rank_size = 20, seed = 0):
    
    conf = (alpha*training_set) #Here we set our confidence matrix to stay sparse.
    
    # Get the size of our original ratings matrix, m x n
    num_user = conf.shape[0] 
    num_item = conf.shape[1] 
    
    # initialize our X/Y feature vectors randomly with a set seed
    rstate = np.random.RandomState(seed)
    
    # Random numbers in a m x rank shape
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size)))
    
    # Normally this would be rank x n but we can # transpose at the end. Makes calculation more simple.
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size)))  
    
    X_eye = sparse.eye(num_user) 
    Y_eye = sparse.eye(num_item) 
    
    # Our regularization term lambda*I.
    lambda_eye = lambda_val * sparse.eye(rank_size)  # We can compute this before iteration starts. 

    # Begin iterations
    # Iterate back and forth between solving X given fixed Y and vice versa
    for iter_step in range(iterations):
        #Compute yTy and xTx at beginning of each iteration to save computing time 
        yTy = Y.T.dot(Y) 
        xTx = X.T.dot(X)
    
        #Begin iteration to solve for X based on fixed Y 
        # Grab user row from confidence matrix and convert to dense
        for u in range(num_user): 
            conf_samp = conf[u,:].toarray() 
    
            # Create binarized preference vector 
            pref = conf_samp.copy() 
            pref[pref != 0] = 1
    
            # Get Cu - I term, don’t need to subtract 1 since we never added it 
            CuI = sparse.diags(conf_samp, [0])
            #Cu = CuI + Y_eye
    
            # This is the yT(Cu-I)Y term  
            yTCuIY = Y.T.dot(CuI).dot(Y) 
            # This is the yTCuPu term, where we add the eye back in Cu - I + I
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambdaI)^-1)yTCuPu 
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 


    # Begin iteration to solve for Y based on fixed X 
    for i in range(num_item): 
            # transpose to get it in row format and convert to dense
            conf_samp = conf[:,i].T.toarray()
            # Create binarized preference vector 
            pref = conf_samp.copy() 
            pref[pref != 0] = 1 
            # Get Ci - I term, don’t need to subtract 1 since we never added it 
            CiI = sparse.diags(conf_samp, [0]) 
    
            # This is the xT(Cu-I)X term
            xTCiIX = X.T.dot(CiI).dot(X) 
            # This is the xTCiPi term 
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) 
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambdaI)^-1)xTCiPi
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi) 
     
    # End iterations 
    return X, Y.T 
# Transpose at the end to make up for not being transposed at the beginning. 
# Y needs to be rank x n. Keep these as separate matrices for scale reasons.    

user_vecs, item_vecs = implicit_weighted_ALS(product_train, lambda_val = 0.1, alpha = 15, iterations = 1, rank_size = 20)

user_vecs[0,:].dot(item_vecs).toarray()[0,:5]

array([5.24299219e-02, 9.85570586e-05, 1.25361284e-02, 3.89815739e-02,
       7.83943702e-03])

 

Speeding up the ALS

#pip install implicit

import implicit

# The implicit library expects data as a item-user matrix so we
# create two matricies, one for fitting the model (item-user) 
# and one for recommendations (user-item)
#sparse_item_user = sparse.csr_matrix((grouped_purchased['OrderQuantity'].astype(float), (grouped_purchased['ProductKey'], grouped_purchased['CustomerKey'])))
#sparse_user_item = sparse.csr_matrix((grouped_purchased['OrderQuantity'].astype(float), (grouped_purchased['CustomerKey'], grouped_purchased['ProductKey'])))

# Initialize the als model and fit it using the sparse item-user matrix
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (product_train * alpha_val).astype('double')

# Fit the model
model.fit(data_conf)

Evaluating the Recommender System

from sklearn import metrics

def auc_score(predictions, test): #This function outputs the area under the curve using sklearn's metrics parameters
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions) 
    return metrics.auc(fpr, tpr)   

def calc_mean_auc(training_set, altered_users, predictions, test_set): #This function will calculate the mean AUC by user for any user that had their user-item matrix altered. parameters:

    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = []  # To store popular AUC scores
    # Get sum of item iteractions to find most popular  
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1)  
    item_vecs = predictions[1] 
    for user in altered_users: # Iterate through each user that had an item altered
        # Get the training set row 
        training_row = training_set[user,:].toarray().reshape(-1)
        # Find where the interaction had not yet occurred
        zero_inds = np.where(training_row == 0)  
        # Get the predicted values based on our user/item vectors 
        user_vec = predictions[0][user,:] 
        # Get only the items that were originally zero 
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1) 
        # Select all ratings from the MF prediction for this user that originally had no iteraction 
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data 
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items 
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store 
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score 
    # End users iteration

    return float("%.3f"%np.mean(store_auc)), float("%.3f"%np.mean(popularity_auc)) # Return the mean AUC rounded to three decimal places for both test and popularity benchmark

user_vecs.shape

(18484, 20)

item_vecs.shape

(20, 158)

calc_mean_auc(product_train, product_users_altered, 
              [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs)], product_test)

(0.643, 0.836)

Testing the Recommender System

customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix 
products_arr = np.array(products) # Array of product IDs from the ratings matrix

Previous purchase by the customer

def get_items_purchased(customer_id, mf_train, customers_list, products_list, item_lookup):
    cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of our customer id
    purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items
    prod_codes = products_list[purchased_ind] # Get the stock codes for our purchased items
    return item_lookup.loc[item_lookup.ProductKey.isin(prod_codes)]

customers_arr[:5]

array([11000, 11001, 11002, 11003, 11004], dtype=int64)

Enter the customer key to get items purchased

get_items_purchased(11900, product_train, customers_arr, products_arr, item_lookup)

	ProductKey 	Product_Description
1 	346 	Mountain-100 Silver, 44
5432 	353 	Mountain-200 Silver, 38
5455 	222 	Sport-100 Helmet, Blue
5459 	573 	Touring-1000 Blue, 46
5504 	489 	Short-Sleeve Classic Jersey, M
Recommending Products to a Customer

from sklearn.preprocessing import MaxAbsScaler

def rec_items(customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup, num_items = 10):
    cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id
    pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
    pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
    rec_vector = user_vecs[cust_ind,:].dot(item_vecs).toarray() # Get dot product of user vector and all item vectors
    # Scale this recommendation vector between 0 and 1
    max_abs = MaxAbsScaler()
    rec_vector_scaled = max_abs.fit_transform(rec_vector.reshape(-1,1))[:,0] 
    recommend_vector = pref_vec*rec_vector_scaled 
    # Items already purchased have their recommendation multiplied by zero
    product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order of best recommendations
   
    rec_list = [] # start empty list to store items
    for index in product_idx:
        code = item_list[index]
        rec_list.append([code, item_lookup.Product_Description.loc[item_lookup.ProductKey == code].iloc[0]]) 
        # Append our descriptions to the list
    codes = [item[0] for item in rec_list]
    descriptions = [item[1] for item in rec_list]
    final_frame = pd.DataFrame({'ProductKey': codes, 'Product_Description': descriptions}) # Create a dataframe 
    return final_frame[['ProductKey', 'Product_Description']] # Switch order of columns around

Enter the customer ID to get the list of products recommended

rec_items(11900, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup,
                       num_items = 10)

	ProductKey 	Product_Description
0 	477 	Water Bottle - 30 oz.
1 	479 	Road Bottle Cage
2 	478 	Mountain Bottle Cage
3 	310 	Road-150 Red, 62
4 	311 	Road-150 Red, 44
5 	487 	Hydration Pack - 70 oz.
6 	225 	AWC Logo Cap
7 	386 	Road-550-W Yellow, 42
8 	463 	Half-Finger Gloves, S
9 	363 	Mountain-200 Black, 46
Enter Customer ID to get list of recommendations

Customer_id = int(input('The ID of the customer is: '))
Reco_number = int(input('Number of recommendations: '))

The ID of the customer is: 11000
Number of recommendations: 10

List of items purchased vs recommended

print('Following is the list of items purchased by Customer no.', Customer_id)
get_items_purchased(Customer_id, product_train, customers_arr, products_arr, item_lookup)

Following is the list of items purchased by Customer no. 11000

	ProductKey 	Product_Description
5432 	353 	Mountain-200 Silver, 38
5442 	214 	Sport-100 Helmet, Red
5457 	541 	Touring Tire
5458 	530 	Touring Tire Tube
5459 	573 	Touring-1000 Blue, 46
5476 	485 	Fender Set - Mountain

print('Following is the list of items recommended to Customer no.', Customer_id)
rec_items(Customer_id, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup,
                       num_items = Reco_number)

Following is the list of items recommended to Customer no. 11000

	ProductKey 	Product_Description
0 	477 	Water Bottle - 30 oz.
1 	225 	AWC Logo Cap
2 	478 	Mountain Bottle Cage
3 	222 	Sport-100 Helmet, Blue
4 	479 	Road Bottle Cage
5 	487 	Hydration Pack - 70 oz.
6 	491 	Short-Sleeve Classic Jersey, XL
7 	361 	Mountain-200 Black, 42
8 	310 	Road-150 Red, 62
9 	488 	Short-Sleeve Classic Jersey, S
