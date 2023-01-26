# Metacritic webcrawler and Movie recomendation system 
---

- This program will crawl through the metacritic.com website and will collect the details of top 500 movies(directors , genre)etc. 
- Then use the data to constuct dataframes and dictionaries for easy retrival of required data. 
- Compute the cosine distance between movies(based on genre) and then provide recomendation to users"


#### Import Libraries for the Task


```python
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import requests
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import copy
from itertools import chain
from numpy import dot
from numpy.linalg import norm
# url = 'https://www.metacritic.com/browse/movies/score/metascore/all/filtered?sort=desc'

# user_agent = {'User-agent': 'Mozilla/5.0'}
# response = requests.get(url, headers = user_agent)

# soup = BeautifulSoup(response.text, 'html.parser')
# k = soup.find_all("a", {"class": "title"})
```

## URL CRAWLER

Crawls through meta critic site and fetches all Movie names and their main pages link


```python
MOVIENAME = []
MOVIELINK = []

url = 'https://www.metacritic.com/browse/movies/score/metascore/all/filtered?sort=desc'


```

### Links retrival from subpages

retrives the link from all the sub pages , since only 100 results appear on one page


```python
from tqdm import tqdm

for i in tqdm(range(5)):
    if(i==0):
        url = url
    else:
        url = url + f"&page={i}"
    user_agent = {'User-agent': 'Mozilla/5.0'}
    response = requests.get(url, headers = user_agent)

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all("a", {"class": "title"})
    
    for k in tqdm(links):
        MOVIENAME.append(k.text)
        MOVIELINK.append( "https://www.metacritic.com" +  k['href'])
```

      0%|                                                                                   | 0/5 [00:00<?, ?it/s]
    100%|███████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 180400.17it/s][A
     20%|███████████████                                                            | 1/5 [00:01<00:07,  1.94s/it]
    100%|███████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 244280.96it/s][A
     40%|██████████████████████████████                                             | 2/5 [00:03<00:05,  1.78s/it]
    0it [00:00, ?it/s][A
     60%|█████████████████████████████████████████████                              | 3/5 [00:04<00:02,  1.49s/it]
    0it [00:00, ?it/s][A
     80%|████████████████████████████████████████████████████████████               | 4/5 [00:06<00:01,  1.50s/it]
    0it [00:00, ?it/s][A
    100%|███████████████████████████████████████████████████████████████████████████| 5/5 [00:07<00:00,  1.49s/it]


### Save the links as csv 

save the fetched links as csv for book keeping, we can use this later to collecct other data


```python
import pandas as pd
from tqdm import tqdm
MovieLinksDF = pd.DataFrame()
MovieLinksDF["movie"] = MOVIENAME
MovieLinksDF["link"] = MOVIELINK

MovieLinksDF.to_csv("MovieLinks.csv")
```


```python
MovieLinksDF
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Godfather</td>
      <td>https://www.metacritic.com/movie/the-godfather</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Citizen Kane</td>
      <td>https://www.metacritic.com/movie/citizen-kane</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rear Window</td>
      <td>https://www.metacritic.com/movie/rear-window</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Casablanca</td>
      <td>https://www.metacritic.com/movie/casablanca</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boyhood</td>
      <td>https://www.metacritic.com/movie/boyhood</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>The African Queen</td>
      <td>https://www.metacritic.com/movie/the-african-q...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>The Triplets of Belleville</td>
      <td>https://www.metacritic.com/movie/the-triplets-...</td>
    </tr>
    <tr>
      <th>197</th>
      <td>The Queen</td>
      <td>https://www.metacritic.com/movie/the-queen</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Flee</td>
      <td>https://www.metacritic.com/movie/flee</td>
    </tr>
    <tr>
      <th>199</th>
      <td>All Quiet on the Western Front</td>
      <td>https://www.metacritic.com/movie/all-quiet-on-...</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>



## Web Crawler

using the links collected for each movies, navigate to the URL of the metacritic site and we could collect the details such as director and genre from the main page. The logic is written to check for missing values and also to accomodate multiple values in a given category


```python
import time
import random
DIRECTORSLIST = []
GENRELIST = []

user_agent_list = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'
    # Firefox
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0'
]

def get_random_agent():
    return random.choice(user_agent_list)

for index, row in tqdm(MovieLinksDF.iterrows()):
    # time.sleep(2)
    url = row['link']
    print("URL : " , url , "movie : " , row['movie'])
    
    user_agent = {'User-agent':  get_random_agent()}
    response = requests.get(url, headers = user_agent)

    if(response.status_code != 200):
        print(" Error in Response : ",response.status_code )
        continue;
    else:
        print("Response Received")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    directors = soup.find_all("div", {"class": "director"})
    directors_a = directors[0].find_all('a')
    
    minDirList = []
    if(len(directors_a) == 0):
        minDirList.append("null")
    else:
        for direc in directors_a:
            minDirList.append(direc.span.text)
    DIRECTORSLIST.append(minDirList)

    genres = soup.find_all("div", {"class": "genres"})
    genreList = genres[0].find_all('span')[1].find_all('span')

    minGenre = []
    if(len(genreList) == 0):
        minDirList.append("null")
    else:  
        for g in genreList:
            minGenre.append(g.text)
    GENRELIST.append(minGenre)
```


```python
user_agent = {'User-agent': 'Mozilla/5.0'}
response = requests.get('https://www.metacritic.com/movie/citizen-kane', headers = user_agent)
print(response)
# soup = BeautifulSoup(response.text, 'html.parser')
# directors = soup.find_all("div", {"class": "director"})
```

    <Response [403]>



```python
import copy

DIRECTORSLIST_copy = copy.deepcopy(DIRECTORSLIST)
GENRELIST_copy = copy.deepcopy(GENRELIST)
```

k[0].text## Add the values to the Dataframe

Add the fetched information to the main dataframe for storage purpose

Note : The dataframe here is used only for temporarly storing the values. The calulation are done using the dictionary datastructure as mentioned in the document


```python
MovieLinksDF["director"] = DIRECTORSLIST
MovieLinksDF["genre"] = GENRELIST
```

#### Save the object for Future access

Dump the dataframe as pickle object


```python
MovieLinksDF.to_pickle('moviedb.pkl')
```

### CSV

Dump the dataframe as csv as instructed


```python
MovieLinksDF
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Godfather</td>
      <td>https://www.metacritic.com/movie/the-godfather</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Citizen Kane</td>
      <td>https://www.metacritic.com/movie/citizen-kane</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rear Window</td>
      <td>https://www.metacritic.com/movie/rear-window</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Casablanca</td>
      <td>https://www.metacritic.com/movie/casablanca</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boyhood</td>
      <td>https://www.metacritic.com/movie/boyhood</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>The African Queen</td>
      <td>https://www.metacritic.com/movie/the-african-q...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>The Triplets of Belleville</td>
      <td>https://www.metacritic.com/movie/the-triplets-...</td>
    </tr>
    <tr>
      <th>197</th>
      <td>The Queen</td>
      <td>https://www.metacritic.com/movie/the-queen</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Flee</td>
      <td>https://www.metacritic.com/movie/flee</td>
    </tr>
    <tr>
      <th>199</th>
      <td>All Quiet on the Western Front</td>
      <td>https://www.metacritic.com/movie/all-quiet-on-...</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>




```python
MovieLinksDF.to_csv('ungoyalla_movies.csv', quotechar='"')
```


```python
import pandas as pd
MovieLinksDF = pd.read_pickle('moviedb.pkl')
```

# Creating Dictionary 
---

We have created a dictionary with multiple layers such that movie name is key and for the next level director name is key
for Multiple director movies, we have added multiple keys (using directors ) on the inner level with same genre


```python
MovieLinksDF
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie</th>
      <th>link</th>
      <th>director</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Godfather</td>
      <td>https://www.metacritic.com/movie/the-godfather</td>
      <td>[Francis Ford Coppola]</td>
      <td>[Drama, Thriller, Crime]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Citizen Kane</td>
      <td>https://www.metacritic.com/movie/citizen-kane</td>
      <td>[Orson Welles]</td>
      <td>[Drama, Mystery]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rear Window</td>
      <td>https://www.metacritic.com/movie/rear-window</td>
      <td>[Alfred Hitchcock]</td>
      <td>[Mystery, Thriller]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Casablanca</td>
      <td>https://www.metacritic.com/movie/casablanca</td>
      <td>[Michael Curtiz]</td>
      <td>[Drama, Romance, War]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boyhood</td>
      <td>https://www.metacritic.com/movie/boyhood</td>
      <td>[Richard Linklater]</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Fateless</td>
      <td>https://www.metacritic.com/movie/fateless</td>
      <td>[Lajos Koltai]</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Happy as Lazzaro</td>
      <td>https://www.metacritic.com/movie/happy-as-lazzaro</td>
      <td>[Alice Rohrwacher]</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>497</th>
      <td>The Fog of War: Eleven Lessons from the Life o...</td>
      <td>https://www.metacritic.com/movie/the-fog-of-wa...</td>
      <td>[Errol Morris]</td>
      <td>[War, Documentary]</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Uncle Boonmee Who Can Recall His Past Lives</td>
      <td>https://www.metacritic.com/movie/uncle-boonmee...</td>
      <td>[Apichatpong Weerasethakul]</td>
      <td>[Fantasy, Comedy]</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Brokeback Mountain</td>
      <td>https://www.metacritic.com/movie/brokeback-mou...</td>
      <td>[Ang Lee]</td>
      <td>[Drama, Romance, Western]</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>
</div>




```python
MOVIEDICT = {}


for index, rows in MovieLinksDF.iterrows():
    dirList = rows['director']
    level2Dict = {}
    for d in dirList:
        level2Dict[d] = rows['genre']
    MOVIEDICT[rows['movie']] = level2Dict
```

### Directors Dict

this dict is similar to the Movie dict, but the main key will be the director. Each director will have their list of movies and genres as the sublevel dictionary.


further, we will also create a dict called `genre_dict` which will be an histogram of all the genres directed by the director and they are maintained in ascending order. This willbe highly usefull in  building the final comparision and cosine distance


```python
DIRECTORDICT = {}

for index, rows in MovieLinksDF.iterrows():
    dirList = rows['director']
    level2Dict = {}
    for d in dirList:
        if d not in DIRECTORDICT:
            DIRECTORDICT[d] = {}
            DIRECTORDICT[d]["movies"] = []
            DIRECTORDICT[d]["genres"] = []
            DIRECTORDICT[d]["movies"].append(rows['movie'])
            DIRECTORDICT[d]["genres"].append(rows['genre']) 
        else:
            DIRECTORDICT[d]["movies"].append(rows['movie'])
            DIRECTORDICT[d]["genres"].append(rows['genre']) 

from itertools import chain
for key in DIRECTORDICT:
    DIRECTORDICT[key]['genres'] = list(chain.from_iterable(DIRECTORDICT[key]['genres'] ))
    uniq = list(set(DIRECTORDICT[key]['genres']))
    genreDict = {}
    for g in uniq:
        genreDict[g] =  DIRECTORDICT[key]['genres'].count(g)
    DIRECTORDICT[key]['genre_dict'] = genreDict
    
    DIRECTORDICT[key]['genre_dict'] = {k: v for k, v in sorted(DIRECTORDICT[key]['genre_dict'].items(), key=lambda item: item[1],reverse=True)}
```


```python

DIRECTORDICT['Steven Spielberg']
```




    {'movies': ["Schindler's List",
      'E.T. The Extra-Terrestrial',
      'Saving Private Ryan',
      'Close Encounters of the Third Kind',
      'Jaws'],
     'genres': ['Biography',
      'Drama',
      'History',
      'War',
      'Adventure',
      'Sci-Fi',
      'Drama',
      'Fantasy',
      'Family',
      'Action',
      'Drama',
      'War',
      'Adventure',
      'Sci-Fi',
      'Drama',
      'Adventure',
      'Thriller',
      'Horror'],
     'genre_dict': {'Drama': 4,
      'Adventure': 3,
      'Sci-Fi': 2,
      'War': 2,
      'Family': 1,
      'Thriller': 1,
      'Fantasy': 1,
      'History': 1,
      'Horror': 1,
      'Action': 1,
      'Biography': 1}}



## compare Director Function
---

- Checks whether the director name exists in the dictionary ( case insensitive custom coded search )
- Obtains the information from director dict
- The prints the movies list and the `genre_dict` which is obtained from the previous section


#### Cosine Similarity 

- Compute the common genres between two directors using set and list method
- Obtain the score array for each director by looping though the common genres.
- if the common genre is not present, then the score is made zero
- now, the cosine dist is computed by `a.b/(||a||||b||)`


```python
from numpy import dot
from numpy.linalg import norm


def compareDirectors(a,b):
    found = 0
    keyVal  = ''
    for key in DIRECTORDICT:
        if(a.lower() == key.lower()):
            found = 1
            keyVal = key
            break;
    if(not found):
        print("Director ", a , " not found ")
        return
    
    printDirectors(a)
    
    dir_a = DIRECTORDICT[keyVal]
    
    found = 0
    keyVal  = ''
    for key in DIRECTORDICT:
        if(b.lower() == key.lower()):
            found = 1
            keyVal = key
            break;
    if(not found):
        print("Director ", b , " not found ")
        return
    
    printDirectors(b)
    
    
    dir_b = DIRECTORDICT[keyVal]
    
    total = dir_a['genres'] + dir_b['genres']
    # print(dir_a['genres'])
    # print(dir_b['genres'])
    genreTotal = list(set(total))
    
    # print(genreTotal)
    # Genrecommon = list(set(dir_a['genres'] + dir_b['genres']))
    # print(Genrecommon)
    score_a = np.zeros(len(genreTotal))
    score_b = np.zeros(len(genreTotal))
    
    for i,g in enumerate(genreTotal):
        if g in dir_a['genre_dict']:
            score_a[i] = dir_a['genre_dict'][g]
        else:
            score_a[i] = 0
    
    for i,g in enumerate(genreTotal):
        if g in dir_b['genre_dict']:
            score_b[i] = dir_b['genre_dict'][g]
        else:
            score_b[i] = 0
    
    
    # print(score_a)
    # print(score_b)
    cos_sim = dot(score_a, score_b)/(norm(score_a)*norm(score_b))
    
    print("Cosine Similarity :" ,cos_sim)
    
```

### Print Director Details 

function which prints the details of the director from the database 


```python
def printDirectors(director):
    found = 0
    keyVal  = ''
    for key in DIRECTORDICT:
        if(director.lower() == key.lower()):
            found = 1
            keyVal = key
            break;
    if(not found):
        print("Director ", director , " not found ")
        return
    
    directorObj = DIRECTORDICT[keyVal]
    
    print(f"Director {keyVal}  has directed ",end="")
    for m in directorObj['movies']:
        print(f'{m}',end=", ")
    print('')
    
    
    print(f"His most directed genres are ",end="")
    
    for k,v in directorObj['genre_dict'].items():
        print(f"{k} : {v}",end=", ")
    print("")
    
    print('')
```


```python
printDirectors("Steven Spielberg")
```

    Director Steven Spielberg  has directed Schindler's List, E.T. The Extra-Terrestrial, Saving Private Ryan, Close Encounters of the Third Kind, Jaws, 
    His most directed genres are Drama : 4, Adventure : 3, Sci-Fi : 2, War : 2, Family : 1, Thriller : 1, Fantasy : 1, History : 1, Horror : 1, Action : 1, Biography : 1, 
    


### Movie Detail

function identifies the details of the movie from the movie dict and then displays the details


```python
def findMovieDetail(movie):
    found = 0
    keyVal  = ''
    for key in MOVIEDICT:
        if(movie.lower() == key.lower()):
            found = 1
            keyVal = key
            break;
    if(not found):
        print("Movie ", movie , " not found ")
        return
    
    movieObj = MOVIEDICT[keyVal]
    
    print("The director of the movie is ",end="")
    for direc in movieObj.keys():
        print(f"{direc}",end=", ")
    
    print("")
    
    print("The genre of the movie is : ",end="" )
    for direc in movieObj.keys():
        for genre in movieObj[direc]:
            print(f"{genre}",end=", ")
```


```python
findMovieDetail("saving PRIVATE ryan")
```

    The director of the movie is Steven Spielberg, 
    The genre of the movie is : Action, Drama, War, 

### Example output of Compare function


```python
compareDirectors('Steve McQueen','Steven Spielberg')
```

    Director Steve McQueen  has directed 12 Years a Slave, Small Axe: Lovers Rock, Small Axe: Mangrove, Small Axe: Education, 
    His most directed genres are Drama : 4, History : 1, Biography : 1, 
    
    Director Steven Spielberg  has directed Schindler's List, E.T. The Extra-Terrestrial, Saving Private Ryan, Close Encounters of the Third Kind, Jaws, 
    His most directed genres are Drama : 4, Adventure : 3, Sci-Fi : 2, War : 2, Family : 1, Thriller : 1, Fantasy : 1, History : 1, Horror : 1, Action : 1, Biography : 1, 
    
    Cosine Similarity : 0.6708203932499369


### MAIN CODE


```python
while True:
    print("\nWhat do you want to check on Metacritic? (Please choose ‘movie’, ‘director’, or ‘comparision’)")
    com = input('input:')
    
    if(com.lower() == 'movie'):
        print("What movie do you want to check?")
        com = input('input: ')
        findMovieDetail(com)
    elif(com.lower() == 'director'):
        print("Who do you want to check?")
        com = input('input: ')
        printDirectors(com)
    elif(com.lower() == 'comparision'):
        print("Who do you want to check?")
        com = input('input: ')
        com1 = input('input: ')
        compareDirectors(com,com1)
    else:
        break;
```

    
    What do you want to check on Metacritic? (Please choose ‘movie’, ‘director’, or ‘comparision’)


    input: MOVIE


    What movie do you want to check?


    input:  saving private ryan


    The director of the movie is Steven Spielberg, 
    The genre of the movie is : Action, Drama, War, 
    What do you want to check on Metacritic? (Please choose ‘movie’, ‘director’, or ‘comparision’)



