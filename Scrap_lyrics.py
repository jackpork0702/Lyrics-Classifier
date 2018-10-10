from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import re
import random
import lyricsgenius as genius  # calling the API
import pprint


# country singer list
def get_country_list(URL):
    page = requests.get(URL)
    html = BeautifulSoup(page.text, "html.parser")
    h = html.find_all("h3", class_="c-list__title t-bold")
    names = []
    new_names = []
    for name in h:
        try:
            names.append(name.get_text().split())

        except:
            pass
    for n in names:
        n = ' '.join(n)
        new_names.append(n)

    return new_names


# getting Hip-Hop Musician List from ranker
def get_rap_list(URL):
    page = requests.get(URL)
    html = BeautifulSoup(page.text, "html.parser")
    h = html.find_all("div", class_="listItem__data")
    names = []
    for href in h:
        try:
            name = href.find('a').contents[0]
            names.append(name)
        except:
            pass

    return names

# getting R&B Musician List from ranker
def get_rmb_list(URL):
    page = requests.get(URL)
    html = BeautifulSoup(page.text, "html.parser")
    h = html.find_all("div", class_="listItem__data")
    names = []
    for href in h:
        try:
            name = href.find('a').contents[0]
            names.append(name)
        except:
            pass

    return names

def get_rock_list(URL):
    page = requests.get(URL)
    html = BeautifulSoup(page.text, "html.parser")
    h = html.find_all("div", class_="listItem__data")
    names = []
    for href in h:
        try:
            name = href.find('a').contents[0]
            names.append(name)
        except:
            pass

    return names



def get_lyrics(artist_names, max_songs=None):  # max_song means how songs  you wanna get from each artist
    api = genius.Genius('4-SobUcjO9vPIAerJeZDGW_2M4l6xGcyQxXWa5HyXclyI92HcHQdMBqoqewAPfxC')  # my token
    artist = api.search_artist(artist_names, max_songs=max_songs,take_first_result=True)

    #   artist=api.search_artist(artist_names[0],max_songs=max_songs,take_first_result=True)#in this case i only type fisrt artist in artist_names
    aux = api.save_artists(artist, overwrite=True)  # param for save_list


def main():
    url_rap = "https://www.ranker.com/crowdranked-list/the-greatest-rappers-of-all-time"
    url_country = 'https://www.rollingstone.com/music/music-country-lists/100-greatest-country-artists-of-all-time-195775/gary-stewart-194252/'
    url_rock='https://www.ranker.com/crowdranked-list/the-best-rock-bands-of-all-time'
    url_sup_rock='https://www.ranker.com/crowdranked-list/the-best-rock-bands-of-all-time?page=2'
    url_rmb='https://www.ranker.com/list/best-r-and-b-artists-and-bands/reference'
    rap_sup_list = ['El-P', 'Freddie Gibbs', 'Mick Jenkins', 'Big K.R.I.T', 'Styles P', 'Fat Joe', 'Ol Dirty Bastard',
                     'Bun B', 'Macklemore', 'Rick Ross', 'Young Jeezy', 'Beastie Boys', 'Danny Brown', 'Lil Kim',
                     'Biz Markie', 'Ab-Soul', 'Fabolous', 'E-40', 'Masta Ace', 'Xzibit', 'Nicki Minaj', 'Queen Latifah',
                     'MC Lyte', 'Kool Moe Dee', 'Ice-T', 'Canibus', 'Jadakiss', 'Run DMC', 'Aesop Rock',
                     'Jay Electronica', 'Mac Miller']
    country_sup_list=['Keith Whitley','Don Williams','Ronnie Milsap','Jim Reeves','Don Gibson','John Anderson',
                      'Johnny Horton','Emmylou Harris','Hank Thompson','Doug Sahm','Webb Pierce','Vince Gill','Faron Young',
                      'Marty Robbins','Billy Joe Shaver','Roy Acuff','Tanya Tucker','Guy Clark','Connie Smith','Vern Gosdin',
                      'Dwight Yoakam','Jessi Colter','Merle Travis','Lee Ann Womack','Asleep at the Wheel','Marty Stuart',
                      'Patty Loveless','Rosanne Cash','Alabama','Taylor Swift','Statler Brothers','Lynn Anderson','Townes Van Zandt',
                      'Steve Earle','Eric Church','Bill Anderson','Jamey Johnson','The Judds','Tim McGraw','Crystal Gayle','Lucinda Williams',
                      'Chris LeDoux','Jerry Jeff Walker','Alison Krauss','Brooks & Dunn','Toby Keith','Brad Paisley','Keith Urban','Carrie Underwood',
                      'John Denver']
    rmb_sup_list=['Chris Brown','R. Kelly','B.B. King','Rick James','Chaka Khan','Patti LaBelle','Frank Ocean','Commodores','The Weeknd','Anita Baker',
                  'John Legend','Donny Hathaway','Amy Winehouse','Dionne Warwick','Stephanie Mills','Jennifer Hudson','The Whispers','DAngelo','The Pointer Sisters',
                  'Maxwell','The Gap Band','Teena Marie','Rihanna','The Staple Singers','Sheila E.','Monica','Sam & Dave','Billy Ocean','Ohio Players','Keith Sweat',
                  'Ne-Yo','Jodeci','Jamie Foxx','Morris Day & the Time','En Vogue','The S.O.S. Band','Adele','Ginuwine','Fantasia','Tony! Toni! Toné!',
                  'India.Arie','Robin Thicke','Dru Hill','Ciara','Brian McKnight','Bell Biv DeVoe','Patrice Rushen','Angie Stone']
    artist_names = get_rock_list(url_sup_rock)
    print(artist_names)
    print(len(artist_names))

    for name in artist_names:
        try:
            get_lyrics(name, max_songs=1000)

        except:
            try:
                get_lyrics(name, max_songs=1000)
            except:
                pass

if __name__ == '__main__':
    main()