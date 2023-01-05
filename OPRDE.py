# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
""" the original """
# Version 1.7; September 2022 added PRDE
# Version 1.6; September 2021 added PRSH
# Version 1.5; 02 Jan 2021 -- was meant to be the final version before switch to BSE2.x, but that didn't happen :-)
# Version 1.4; 26 Oct 2020 -- change to Python 3.x
# Version 1.3; July 21st, 2018 (Python 2.x)
# Version 1.2; November 17th, 2012 (Python 2.x)
#
# Copyright (c) 2012-2022, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1 (will add variable quantities later)
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

"""in this version opinions are updated as follows
we have PRDE and OPRDE
        (a) Inputs from the environment (event and global) opinions are updated for all traders in each itration
        (b) local opinion is the  ( weighted sum of neighbours opinions) is updated for each trader randomly
        (c) this is the code used for the ICAART paper
        """

import sys
import math
import random
import time as chrono
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from methods import Get_Attention
from warnings import simplefilter
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot
# from statsmodels.tsa.ar_model import AutoReg
# from sklearn.metrics import mean_squared_error
# from math import sqrt


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# a bunch of system constants (globals)
bse_sys_minprice = 1  # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500  # maximum price in the system, in cents/pennies
bse_sys_tape_len = 20  # maximum number of events recorded on tape (most recent are retained)
bse_sys_blotter_len = 20  # maximum number of events recorded on tape (most recent are retained)
bse_sys_PRSH_landscape_mapper = True  # boolean that says whether PRSH is SHC or landscape mapper todo fix this to PRZI

# ticksize should be a param of an exchange (so different exchanges have different ticksizes)
# todo: change this, so ticksize no longer global. NB could do same for tape_len and blotter_len
ticksize = 1  # minimum change in price, in cents/pennies


# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

    def __init__(self, tid, otype, price, qty, time, qid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.qid = qid  # quote i.d. (unique to each quote)

    def __str__(self):
        return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
               (self.tid, self.otype, self.price, self.qty, self.time, self.qid)


# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None  # most extreme price quoted in this session
        self.session_extreme_bid = None
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        # anonymize a lob, strip out order details, format as a sorted list
        # NB for asks, the sorting should be reversed
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        lob_verbose = False
        # take a list of orders and build a limit-order-book (lob) from it
        # NB the exchange needs to know arrival times and trader-id associated with each order
        # returns lob as a dictionary (i.e., unsorted)
        # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        # add order to the dictionary holding the list of orders
        # either overwrites old order from this trader
        # or dynamically creates new entry in the dictionary
        # so, max of one order per trader per list
        # checks whether length or order list has changed, to distinguish addition/overwrite
        # print('book_add > %s %s' % (order, self.orders))

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        if (self.booktype == 'Bid') and (
                (self.session_extreme_bid is None) or (order.price < self.session_extreme_bid)):
            self.session_extreme_bid = int(order.price)

        # add the order to the book
        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        # delete order from the dictionary holding the orders
        # assumes max of one order per trader per list
        # checks that the Trader ID does actually exist in the dict before deletion
        # print('book_del %s',self.orders)
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        # delete order: when the best bid/ask has been hit, delete it from the book
        # the TraderID of the deleted order is return-value, as counterparty to the trade
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice

                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

    def __init__(self):
        self.bids = Orderbook_half('Bid', bse_sys_minprice)
        self.asks = Orderbook_half('Ask', bse_sys_maxprice)
        self.tape = []
        self.tape_length = 100  # max number of records on tape (so we can do millions of orders without crashing)
        self.quote_id = 0  # unique ID code for each quote accepted onto the book


# Exchange's internal orderbook

class Exchange(Orderbook):

    def add_order(self, order, verbose):
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, verbose):
        # delete a trader's quot/order from the exchange, update all internal records
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order2(self, time, order, verbose):
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None
        [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.qid = qid

        # verbose = True
        if verbose:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid

        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:

                # bid lifts the best ask

                if verbose:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # print("best bid", best_bid)
                # ask hits the best bid
                # verbose = True
                if verbose:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            #verbose = True
            if verbose:
                print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            self.tape.append(transaction_record)
            # print("tape", self.tape)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

            return transaction_record
        else:
            return None

    # Currently tape_dump only writes a list of transactions (ignores cancellations)
    def tape_dump(self, dumpfile, tmode):
        #dumpfile = open(fname, fmode)
        # dumpfile.write('type, time, price\n')
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                dumpfile.write('Trd, %010.3f, %s\n' % (tapeitem['time'], tapeitem['price']))

        if tmode == 'wipe':
            self.tape = []

    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    def publish_lob(self, time, verbose):
        public_data = {}
        public_data['time'] = time
        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'sess_low': self.bids.session_extreme_bid,
                               'n': self.bids.n_orders,
                               'lob': self.bids.lob_anon}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': self.asks.lob_anon}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape
        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
            print('ASK_lob=%s' % public_data['asks']['lob'])
            # print('qid=%d' % self.quote_id)

        return public_data


##################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

    def __init__(self, ttype, tid, balance, params, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.params = params  # parameters/extras associated with this trader-type or individual trader.
        self.blotter = []  # record of trades executed
        self.blotter_length = 100  # maximum length of blotter
        self.orders = []  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.birthtime = time  # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0  # how many trades has this trader done?
        self.lastquote = None  # record of what its last quote was

        self.opinion = None  # or opinion dynamics, trader's overall opinion
        self.opinion_local = 0.0  # "local" opinion, for opinion dynamics -- set in social-network interactions sum1
        self.opinion_global = 0.0  # "global" opinion, for opinion dynamics -- set in response to the LOB input (b)
        self.opinion_event = 0.0  # the event opinion
        self.opinion_comsources = 0.0  # 'communication sources' opinions e.g. soical media or news
        self.opinion_t0 = None  # initial (at time=0) opinion
        # self.opinion_alpha = 1.0  # weight coeff for self opinion
        # self.opinion_gamma = 1.0  # weight coeff for local opinion
        # self.opinion_delta = 1.0  #weight coeff for communication sources opinion
        self.obstinant = 1.0  # how changeable is initial opinion? (obstinant=1 => not at all) (resitence)
        self.attention = 0.5  # how attentive the traders is the environemnt (neighbours and communication sources

    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # todo: CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders = []

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos
    def respond(self, time, lob, trade, verbose):
        return None

    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
        return None


# Trader subclass PRZI (ticker: PRSH)
# added 6 Sep 2022 -- replaces old PRZI and PRZI_SHC, unifying them into one function and also adding PRDE
#
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader -- pronounced "prezzie"
# but with added adaptive strategies, currently either...
#   ++ a k-point Stochastic Hill-Climber (SHC) hence PRZI-SHC,
#      PRZI-SHC pronounced "prezzy-shuck". Ticker symbol PRSH pronounced "purrsh";
# or
#   ++ a simple differential evolution (DE) optimizer with pop_size=k, hence PRZE-DE or PRDE ('purdy")
#
# when optimizer == None then it implements plain-vanilla non-adaptive PRZI, with a fixed strategy-value.

class Trader_PRZI(Trader):

    # how to mutate the strategy values when evolving / hill-climbing
    def mutate_strat(self, s, mode):
        s_min = self.strat_range_min
        s_max = self.strat_range_max
        if mode == 'gauss':
            sdev = 0.05
            newstrat = s
            while newstrat == s:
                newstrat = s + random.gauss(0.0, sdev)
                # truncate to keep within range
                newstrat = max(-1.0, min(1.0, newstrat))
        elif mode == 'uniform_whole_range':
            # draw uniformly from whole range
            newstrat = random.uniform(-1.0, +1.0)
        elif mode == 'uniform_bounded_range':
            # draw uniformly from bounded range
            newstrat = random.uniform(s_min, s_max)
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat

    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = '%s: %s active_strat=[%d]:\n' % (self.tid, self.ttype, self.active_strat)
        for s in range(0, len(self.strats)):
            strat = self.strats[s]
            stratstr = '[%d]: s=%+f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string

    def __init__(self, ttype, tid, balance, params, time):
        # if params == "landscape-mapper" then it generates data for mapping the fitness landscape

        verbose = False

        Trader.__init__(self, ttype, tid, balance, params, time)

        # unpack the params
        if type(params) is dict:
            k = params['k']
            optimizer = params['optimizer']
            s_min = params['strat_min']
            s_max = params['strat_max']
        else:
            optimizer = None
            s_min = 0.0
            s_max = 0.0

        self.optmzr = optimizer  # this determines whether it's PRZI, PRSH, PRDE, or OPRDE
        self.k = k  # number of sampling points (cf number of arms on a multi-armed-bandit, or pop-size)
        self.theta0 = 100  # threshold-function limit value
        self.m = 4  # tangent-function multiplier
        self.strat_wait_time = 7200  # how many secs do we give any one strat before switching?
        self.strat_range_min = s_min  # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = s_max  # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0  # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None  # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time  # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.0 * random.random()  # minimum profit-per-sec difference between strategies that counts
        self.strats = []  # strategies awaiting initialization
        self.pmax = None  # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1, 10))  # multiplier coefficient when estimating p_max
        self.pmin = None  # this trader's estimate of the maximum price the market will bear
        # self.pmin_c_i = pow((random.randint(1, 10)),2)# divider coefficient when estimating p_min to bes set later

        self.mapper_outfile = None
        # differential evolution parameters all in one dictionary
        self.diffevol = {'de_state': 'active_s0',  # initial state: strategy 0 is active (being evaluated)
                         's0_index': self.active_strat,  # s0 starts out as active strat
                         'snew_index': self.k,  # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,  # assigned later
                         'sopinion_index': self.k + 1,
                         'sopinion_stratval': None,  # the value taken from the traders opinion
                         'F': 0.8  # differential weight -- usually between 0 and 2
                         }

        # print("-----------------------------------------------------------------\n diffevol", self.diffevol)
        start_time = time
        profit = 0.0
        profit_per_second = 0
        lut_bid = None
        lut_ask = None

        for s in range(self.k + 1):
            # initialise each of the strategies in sequence: for PRSH, one random seed, then k-1 mutants of that seed
            # for PRDE, use draws from uniform distbn over whole range
            # the (k+1)th strategy is needed to hold s_new in differential evolution; it's not used in SHC.
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                if self.optmzr == 'PRSH':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')  # mutant of strats[0]
                elif self.optmzr == 'PRDE' or self.optmzr == 'OPRDE':
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_bounded_range')
                else:
                    sys.exit('bad self.optmzr when initializing PRZI strategies')

            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})


        if self.optmzr == 'OPRDE':
            strategy = self.opinion
            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})

        if self.params == 'landscape-mapper':
            # replace seed+mutants set of strats with regularly-spaced strategy values over the whole range
            self.strats = []
            strategy_delta = 0.01
            strategy = -1.0
            k = 0
            self.strats = []
            while strategy <= +1.0:
                self.strats.append({'stratval': strategy, 'start_t': start_time,
                                    'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
                k += 1
                strategy += strategy_delta
            self.mapper_outfile = open('landscape_map.csv', 'w')
            self.k = k
            self.strat_eval_time = self.k * self.strat_wait_time

        # verbose = True
        if verbose:
            print("%s\n" % self.strat_str())

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + ticksize  # BSE ticksize is global var
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - ticksize  # BSE ticksize is global var
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            # print('shvr_p=%f; ' % shvr_p)
            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):

            # set parameter values and calculate CDF LUT
            # strat is strategy-value in [-1,+1]
            # t0 and m are constants used in the threshold function
            # dirn is direction: 'buy' or 'sell'
            # pmin and pmax are bounds on discrete-valued price-range

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1 * theta0, min(theta0, x))
                return t

            epsilon = 0.000001  # used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: strat=%f out of range\n' % strat)

            if (dirn != 'buy') and (dirn != 'sell'):
                # out of range
                sys.exit('PRSH FAIL: bad dirn=%s\n' % dirn)

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax %f < pmin %f \n' % (pmax, pmin))

            if verbose:
                print('PRSH calc_cdf_lut: strat=%f dirn=%s pmin=%d pmax=%d\n' % (strat, dirn, pmin, pmax))

            p_range = float(pmax - pmin)

            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the limit-price with probability 1
                print("the shver saheved to the limit price")
                if dirn == 'buy':
                    cdf = [{'price': pmax, 'cum_prob': 1.0}]
                else:  # must be a sell
                    cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                # normalize the price to proportion of its range
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    if dirn == 'buy':
                        cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                    else:  # dirn == 'sell'
                        cal_p = (math.exp(c * (1 - p_r)) - 1.0) / e2cm1
                else:  # self.strat < 0
                    if dirn == 'buy':
                        cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                    else:  # dirn == 'sell'
                        cal_p = 1.0 - ((math.exp(c * (1 - p_r)) - 1.0) / e2cm1)

                if cal_p < 0:
                    cal_p = 0  # just in case

                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p - pmin]['price']  # todo: what does this do?
                cal_p = calp_interval[p - pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})  # todo shouldnt ths be "price" not "p"?
            verbose = False
            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        verbose = False

        if verbose:
            print('t=%.1f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                dummy = 1

            # get extreme limits on price interval
            # lowest price the market will bear

            # print(minprice)
            # trader's individual estimate lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible as defined by exchange
            # minprice = self.pmin  # default assumption
            if self.pmin is None:
                minprice = int(max((limit / self.pmax_c_i - 0.5), minprice))  # in the absence of any other info, guess
                self.pmin = minprice

            elif lob['bids']['sess_low'] is not None:
                if self.pmin > lob['bids']['sess_low']:  # some other trader has quoted lower than I expected
                    minprice = lob['bids']['sess_low']  # so use that as my new estimate of lowest
                    self.pmin = minprice

            # print(lob)

            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax  # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)  # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:  # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']  # so use that as my new estimate of highest
                    self.pmax = maxprice

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),

            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # print(self.tid, "strategy", strat, "Olimit", Olimit, "pmin", self.pmin)
            if otype == 'Bid':
                p_max = int(limit)
                if strat > 0.0:  # orignial if strat > 0.0
                    p_min = minprice  # the prices are going down be down as you can
                else:  # prices are going up move to the limit you need a deal
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    # print(" the chaver price is ", p_shvr)
                    p_min = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * minprice))

                if self.opinion == 0.0:
                    p_min = bse_sys_minprice

                if p_min < bse_sys_minprice:
                    p_min = bse_sys_minprice
                if p_max < p_min:
                    print("P max < p Min")
                    # this should never happen, but just in case it does...
                    p_max = p_min
                # print("Otype", otype, "Olimit", int(Olimit))
                # print("trader", self.tid, "limit", limit, "Olimit", Olimit, "self.pmin", p_min, "self.pmax",p_max)
                #print(self.tid, "pmin", p_min, "pmax", p_max, "strat", strat, "opinion", self.opinion)
                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or \
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, 'buy', p_min,
                                                                             p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:  # otype == 'Ask'
                p_min = int(limit)

                if strat > 0.0:  # prices are going up be high as you can (if strat > 0.0:)
                    p_max = maxprice
                else:  # when s < 0 the prices are going down
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * maxprice))

                if self.opinion == 0.0:
                   p_max = bse_sys_maxprice

                if p_max > bse_sys_maxprice:
                    p_max = bse_sys_maxprice

                if p_max < p_min:
                    # this should never happen, but just in case it does...
                    p_max = p_min

                #print("trader", self.tid, "limit", limit, "Olimit", Olimit, "p_min",p_min, "p max", p_max)
                # print("Otype", otype, "Olimit", int(Olimit))
                #print(self.tid, "pmin", p_min, "pmax", p_max, "strat", strat, "opinion", self.opinion )
                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')

                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, 'sell', p_min,
                                                                             p_max)

                lut = self.strats[self.active_strat]['lut_ask']

            verbose = False
            if verbose:
                print('PRZI strat=%f LUT=%s \n \n' % (strat, lut))
                # useful in debugging: print a table of lut: price and cum_prob, with the discrete derivative (gives PMF).
                last_cprob = 0.0
                for lut_entry in lut['cdf_lut']:
                    cprob = lut_entry['cum_prob']
                    print('%d, %f, %f' % (lut_entry['price'], cprob - last_cprob, cprob))
                    last_cprob = cprob
                print('\n')

                # print ('[LUT print suppressed]')

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        self.strats[self.active_strat]['profit'] += profit
        time_alive = time - self.strats[self.active_strat]['start_t']
        if time_alive > 0:
            profit_per_second = self.strats[self.active_strat]['profit'] / time_alive
            self.strats[self.active_strat]['pps'] = profit_per_second
        else:
            # if it trades at the instant it is born then it would have infinite profit-per-second, which is insane
            # to keep things sensible whne time_alive == 0 we say the profit per second is whatever the actual profit is
            self.strats[self.active_strat]['pps'] = profit

    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self, time, lob, trade, verbose):

        # "PRSH" is a very basic form of stochastic hill-climber (SHC) that's v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other k-1 strats
        # then all counters are reset, and this is repeated indefinitely
        #
        # "PRDE" uses a basic form of Differential Evolution. This maintains a population of at least four strats
        # iterates indefinitely on:
        #       shuffle the set of strats;
        #       name the first four strats s0 to s3;
        #       create S4 =  new_strat=s1+f*(s2-s3);
        #       evaluate fitness of s0 and new_strat;
        #       if (new_strat fitter than s0) then new_strat replaces s0.
        #
        #
        # todo: add in other optimizer algorithms that are cleverer than these
        #  e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        verbose = False

        # first update each strategy's profit-per-second (pps) value -- this is the "fitness" of each strategy
        for s in self.strats:
            # debugging check: make profit be directly proportional to strategy, no noise
            # s['profit'] = 100 * abs(s['stratval'])
            # update pps
            pps_time = time - s['start_t']
            if pps_time > 0:
                s['pps'] = s['profit'] / pps_time
            else:
                s['pps'] = s['profit']

        if self.optmzr == 'PRSH':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                dummy = 1

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.
            # when a trader does a deal, clock is reset; todo check this!!!
            # clock also reset when new a strat is created, obvs. todo check this!!! also check bookkeeping/proft etc

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            # assume that all strats have had long enough, and search for evidence to the contrary
            all_old_enough = True
            for s in self.strats:
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key=lambda k: k['pps'], reverse=True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time - s['start_t'], s['profit'], s['pps']))

                if self.params == 'landscape-mapper':
                    for s in self.strats:
                        self.mapper_outfile.write('time, %f, strat, %f, pps, %f\n' %
                                                  (time, s['stratval'], s['pps']))
                    self.mapper_outfile.flush()
                    sys.exit()

                else:
                    # if the difference between the top two strats is too close to call then flip a coin
                    # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                    best_strat = 0
                    prof_diff = strats_sorted[0]['pps'] - strats_sorted[1]['pps']
                    if abs(prof_diff) < self.profit_epsilon:
                        # they're too close to call, so just flip a coin
                        best_strat = random.randint(0, 1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = strats_sorted[0]
                        strats_sorted[0] = strats_sorted[1]
                        strats_sorted[1] = tmp_strat

                    # the sorted list of strats replaces the existing list
                    self.strats = strats_sorted

                    # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate
                    # record it

                    # now replicate and mutate elite into all the other strats
                    for s in range(1, self.k):  # note range index starts at one not zero
                        self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'], 'gauss')
                        self.strats[s]['start_t'] = time
                        self.strats[s]['profit'] = 0.0
                        self.strats[s]['pps'] = 0.0
                    # and then update (wipe) records for the elite
                    self.strats[0]['start_t'] = time
                    self.strats[0]['profit'] = 0.0
                    self.strats[0]['pps'] = 0.0
                    self.active_strat = 0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time - s['start_t'], s['profit'], s['pps']))
        elif self.optmzr == 'PRDE':
            # simple differential evolution

            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    # we've evaluated s0, so now we need to evaluate s_new
                    self.active_strat = self.diffevol['snew_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # now we've evaluated s_0 and s_new, so we can do DE adaptive step
                    if verbose:
                        print('PRDE trader %s' % self.tid)
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']

                    if verbose:
                        print('DiffEvol: t=%.1f, i_0=%d, i0fit=%f, i_new=%d, i_new_fit=%f' % (
                            time, i_0, fit_0, i_new, fit_new))

                    if fit_new >= fit_0:
                        # new strat did better than old strat0, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))  # create sequential list of strategy-numbers
                    random.shuffle(stratlist)  # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    if verbose:
                        print('DiffEvol: t=%.1f, s0=%d, s1=%d, (s=%+f), s2=%d, (s=%+f), s3=%d, (s=%+f), sNew=%+f' %
                              (time, self.diffevol['s0_index'],
                               s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    sum = 0.0
                    for s in range(self.k):
                        sum += self.strats[s]['stratval']
                    strat_mean = sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    if verbose:
                        print('t=,%.1f, MeanStrat=, %+f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one strategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        if verbose:
                            print('Converged pop: set strategy %d to %+f' % (
                                randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        elif self.optmzr == 'OPRDE':
            # simple differential evolution
            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    # print("we've evaluated s0, so now we need to evaluate s_new")
                    self.active_strat = self.diffevol['snew_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # print("we've evaluated s0, and Snew  so now we need to evaluate s_opinion")
                    self.active_strat = self.diffevol['sopinion_index']

                    if 'B' in self.tid:  # trader is a buyer
                        self.strats[self.active_strat]['stratval'] = math.sin(self.opinion)
                    else:
                        self.strats[self.active_strat]['stratval'] = -math.sin(self.opinion)
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_sopinion'

                elif self.diffevol['de_state'] == 'active_sopinion':
                    # print("now we've evaluated s_0, s_new and s_opinion, so we can do DE adaptive step")
                    if verbose:
                        print('OPRDE trader %s' % self.tid)
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    i_opinion = self.diffevol['sopinion_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']
                    fit_opinion = self.strats[i_opinion]['pps']

                    if verbose:
                        print(
                            'DiffEvol: t=%.1f, i_0=%d, i0fit=%f, i_new=%d, i_new_fit=%f, i_opinion=%d, i_opinion_fit=%f' % (
                                time, i_0, fit_0, i_new, fit_new, i_opinion, fit_opinion))

                    if fit_new >= fit_0 and fit_new >= fit_opinion:
                        # new strat did better than old strat0 and better thatn opinion, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    elif fit_opinion >= fit_0 and fit_opinion >= fit_new:
                        self.strats[i_0]['stratval'] = self.strats[i_opinion]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))  # create sequential list of strategy-numbers
                    random.shuffle(stratlist)  # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new, then s_opinion)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    if 'B' in self.tid:  # trader is a buyer
                        self.strats[self.diffevol['sopinion_index']]['stratval'] = math.sin(self.opinion)
                    else:
                        self.strats[self.diffevol['sopinion_index']]['stratval'] = -math.sin(self.opinion)

                    if verbose:
                        print('DiffEvol: t=%.1f, s0=%d, s1=%d, (s=%+f), s2=%d, (s=%+f), s3=%d, (s=%+f), sNew=%+f[]' %
                              (time, self.diffevol['s0_index'],
                               s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    sum = 0.0
                    for s in range(self.k):
                        sum += self.strats[s]['stratval']
                    strat_mean = sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    if verbose:
                        print('t=,%.1f, MeanStrat=, %+f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one strategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        if verbose:
                            print('Converged pop: set strategy %d to %+f' % (
                                randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        elif self.optmzr is None:
            # this is PRZI -- nonadaptive, no optimizer, nothing to change here.
            pass

        else:
            sys.exit('FAIL: bad value for self.optmzr')


# ########################---trader-types have all been defined now--################


# #########################---Below lies the experiment/test-rig---##################


# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call
def trade_stats(expid, traders, dumpfile, time, lob):
    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('N, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('N, ')

    dumpfile.write('\n')


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, verbose):
    # traders_spec is a list of buyer-specs and a list of seller-specs
    # each spec is (<trader type>, <number of this type of trader>, optionally: <params for this type of trader>)

    def trader_type(robottype, name, parameters):
        balance = 0.00
        time0 = 0
        if robottype == 'PRZI':
            return Trader_PRZI('PRZI', name, balance, parameters, time0)
        elif robottype == 'PRSH':
            return Trader_PRZI('PRSH', name, balance, parameters, time0)
        elif robottype == 'PRDE':
            return Trader_PRZI('PRDE', name, balance, parameters, time0)
        elif robottype == 'OPRDE':
            return Trader_PRZI('OPRDE', name, balance, parameters, time0)
        else:
            sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp

    def unpack_params(trader_params, mapping):
        # unpack the parameters for PRZI-family of strategies
        parameters = None
        if ttype == 'PRSH' or ttype == 'OPRDE' or ttype == 'PRDE' or ttype == 'PRZI':
            # parameters matter...
            if mapping:
                parameters = 'landscape-mapper'
            else:
                # params determines type of optimizer used
                if ttype == 'PRSH':
                    parameters = {'optimizer': 'PRSH', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                elif ttype == 'PRDE':
                    parameters = {'optimizer': 'PRDE', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                elif ttype == 'OPRDE':
                    parameters = {'optimizer': 'OPRDE', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                else:  # ttype=PRZI
                    parameters = {'optimizer': None, 'k': 1,
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}

        return parameters

    landscape_mapping = False  # set to true when mapping fitness landscape (for PRSH etc).

    # the code that follows is a bit of a kludge, needs tidying up.
    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            if len(bs) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(bs[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            if len(ss) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(ss[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}


# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):
    def sysmin_check(price):
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, sched, n, mode, issuetime):
        # does the first schedule range include optional dynamic offset function(s)?
        if len(sched[0]) > 2:
            offsetfn = sched[0][2]
            if callable(offsetfn):
                # same offset for min and max
                offset_min = offsetfn(issuetime)
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(sched[0]) > 3:
                # if second offset function is specfied, that applies only to the max value
                offsetfn = sched[0][3]
                if callable(offsetfn):
                    # this function applies to max
                    offset_max = offsetfn(issuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
        # print("This is the system minimum price", pmin)
        pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
        # print("This is the system maximum price", pmax)
        prange = pmax - pmin
        stepsize = prange / (n - 1)

        halfstep = round(stepsize / 2.0)

        if mode == 'fixed':
            orderprice = pmin + int(i * stepsize)
        elif mode == 'jittered':
            orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif mode == 'random':
            if len(sched) > 1:
                # more than one schedule: choose one equiprobably
                s = random.randint(0, len(sched) - 1)
                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
            orderprice = random.randint(pmin, pmax)
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        orderprice = sysmin_check(sysmax_check(orderprice))

        return orderprice

    def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetime()')
        elif n_traders == 1:
            tstep = interval
        else:
            tstep = interval / (n_traders - 1)
        arrtime = 0
        issuetimes = []
        for t in range(n_traders):
            if mode == 'periodic':
                arrtime = interval
            elif mode == 'drip-fixed':
                arrtime = t * tstep
            elif mode == 'drip-jitter':
                arrtime = t * tstep + tstep * random.random()
            elif mode == 'drip-poisson':
                # poisson requires a bit of extra work
                interarrivaltime = random.expovariate(n_traders / interval)
                arrtime += interarrivaltime
            else:
                sys.exit('FAIL: unknown time-mode in getissuetimes()')
            issuetimes.append(arrtime)

            # at this point, arrtime is the last arrival time
        if fittointerval and ((arrtime > interval) or (arrtime < interval)):
            # generated sum of interarrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for t in range(n_traders):
                issuetimes[t] = interval * (issuetimes[t] / arrtime)
        # optionally randomly shuffle the times
        if shuffle:
            for t in range(n_traders):
                i = (n_traders - 1) - t
                j = random.randint(0, i)
                tmp = issuetimes[i]
                issuetimes[i] = issuetimes[j]
                issuetimes[j] = tmp
        return issuetimes

    def getschedmode(time, os):
        got_one = False
        for sched in os:
            if (sched['from'] <= time) and (time < sched['to']):
                # within the timezone for this schedule
                schedrange = sched['ranges']
                mode = sched['stepmode']
                got_one = True
                break  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
        return (schedrange, mode)

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    # if time >= 43200 and time <=50000:
    #   n_buyers = trader_stats['n_buyers']*10
    #  print("---------------------------------------------------------------")

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)

        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, os['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            # print("order price for the buyer", tname, "is", orderprice)
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)

        # supply side (sellers)
        issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, os['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            # print("order price for the seller", tname, "is", orderprice)
            # print('time %d sellerprice %d' % (time,orderprice))
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)

    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                tname = order.tid
                response = traders[tname].add_order(order, verbose)
                if verbose:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
                    if verbose:
                        print('Cancellations: %s' % cancellations)
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations]


# one session in the market
def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, tdump, dump_all, verbose):
    def dump_strats_frame(time, stratfile, Ofile, trdrs):
        # write one frame of strategy snapshot

        line_str = 't=,%.0f, ' % time
        Oline_str = 't=,%.0f, ' % time

        best_buyer_id = None
        best_buyer_prof = 0
        best_buyer_strat = 0
        best_seller_id = None
        best_seller_prof = 0
        best_seller_strat = 0

        for t in traders:
            trader = trdrs[t]

            # print('PRSH/PRDE recording, t=%s' % trader)
            if trader.ttype == 'PRSH' or trader.ttype == 'PRDE' or trader.ttype == 'OPRDE':
                line_str += 'id=,%s, %s,' % (trader.tid, trader.ttype)

                Oline_str += 'id=,%s, %s,opinion=, %f,' % (trader.tid, trader.ttype, trader.opinion)

                # line_str += 'bal=$,%f, n_trades=,%d, n_strats=,2, ' % (trader.balance, trader.n_trades)

                act_strat = trader.strats[trader.active_strat]['stratval']
                act_prof = trader.strats[trader.active_strat]['pps']

                line_str += 'actvstrat=,%f, ' % act_strat
                line_str += 'actvprof=,%f, ' % act_prof

                if trader.tid[:1] == 'B':
                    # this trader is a buyer
                    if best_buyer_id is None or act_prof > best_buyer_prof:
                        best_buyer_id = trader.tid
                        best_buyer_strat = act_strat
                        best_buyer_prof = act_prof
                elif trader.tid[:1] == 'S':
                    # this trader is a seller
                    if best_seller_id is None or act_prof > best_seller_prof:
                        best_seller_id = trader.tid
                        best_seller_strat = act_strat
                        best_seller_prof = act_prof
                else:
                    # wtf?
                    sys.exit('unknown trader id type in market_session')

        line_str += 'best_B_id=,%s, best_B_prof=,%f, best_B_strat=,%f, ' % \
                    (best_buyer_id, best_buyer_prof, best_buyer_strat)
        line_str += 'best_S_id=,%s, best_S_prof=,%f, best_S_strat=,%f, ' % \
                    (best_seller_id, best_seller_prof, best_seller_strat)
        line_str += '\n'
        Oline_str += '\n'

        stratfile.write(line_str)
        stratfile.flush()

        Ofile.write(Oline_str)
        Ofile.flush()

    def opinion_dynamics(time, timestep, lob, trader_list, A, trans_recording, verbose):

        # opinion_dynamics is called on each iteration of the market_session main loop when there is an order issued
        # (i.e. up to n_traders times per second)
        # this frequency makes sense for global opinion (because LOB could change multiple times per second)
        # but for local opinion a realistic rate of interaction (among humans at least) would be slower than that
        # so pr_activity is probability of *any* local OD activity per call

        # global opinion dynamics: form an opinion by looking at the LOB
        # do we have orders resting at top of both sides of LOB?

        if verbose:
            print("opinion_dynamics is called ")

        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_bid_p is not None and lob_best_ask_p is not None:
            # top of LOB has prices on both sides
            lob_best_bid_q = lob['bids']['lob'][-1][1]  # quantity
            lob_best_ask_q = lob['asks']['lob'][0][1]

            midprice = (lob_best_bid_p + lob_best_ask_p) / 2.0
            total_q = lob_best_bid_q + lob_best_ask_q
            microprice = ((lob_best_bid_p * lob_best_ask_q) + (lob_best_ask_p * lob_best_bid_q)) / total_q

            delta_p_norm = min((microprice - midprice) / midprice, +1.0)
            # print("deta p norm", delta_p_norm)
            verbose = False
            if verbose:
                print('t=%f, OpDyn-global: LOB top: bq=%f bp=%f ap=%f aq=%f; mid=%f micro=%f delta_p_norm=%f' % \
                      (time, lob_best_bid_q, lob_best_bid_p, lob_best_ask_p, lob_best_ask_q, midprice, microprice,
                       delta_p_norm))
        else:
            # top of LOB is blank on one or both sides
            # todo: could be more sophisticated here
            delta_p_norm = 0.0
            # -- if the top of the LOB is empty on one/both sides because market is very thin, or has only just opened
            #    then setting delta_p_norm to zero is a sensible choice.
            # -- but if top of LOB was previously well balanced with "non-thin" quantities and then a big block-order
            #    buy/sell wiped out one side of the LOB, that's a potential sign of major sup/dem imbalance
            #    in which case delta_p_norm could more usefully be set to plus or minus one.

        event_opinion = 0.0
        # Get the event opinion
        # print(trans_recording)

        """ THIS IS EXTEREMLY SLOW 
        slope = None
        if len(trans_recording) > 2:
            trprice_timeS_df = pd.DataFrame(trans_recording)
            trprice_timeS_df = trprice_timeS_df.drop(columns=['type', 'party1', 'party2', 'qty'])
            slope = (trprice_timeS_df['price'].iloc[-1] - trprice_timeS_df['price'].iloc[-2]) / (
                        trprice_timeS_df['time'].iloc[-1] - trprice_timeS_df['time'].iloc[-2])
        """
        slope = None

        trans_recording = trans_recording[-2:]

        if len(trans_recording) >= 2:
            slope = (trans_recording[1]['price'] - trans_recording[0]['price']) / (
                    trans_recording[1]['time'] - trans_recording[0]['time'])

        if slope is not None:
            event_opinion = max(min(slope, 1), -1)
        else:
            event_opinion = 0.0

        # print(event_opinion)

        # local opinion dynamics:
        # todo: here we should add code for established OD models like BC, RA, or RD.
        # in this model the local opinion is the weighted sum of the opinions of the trader neighbours in the network
        # a trader is being chosen at random to update its local opinion using shuffle algorithm
        # Only one trader is going to have a local opinion update at a time

        if int(time) % 60 == 0.0:
        # print("time to cal local opinion", time)
            random_tid = random.choice(list(trader_list.keys()))
            # print(random_tid, "random_tid")
            sum1 = 0.0
            for othertid in trader_list:
                if random_tid != othertid:
                    # sum1 += (A.loc[random_tid, othertid] * trader_list[othertid].opinion)
                    sum1 += (trader_list[othertid].opinion)
            trader_list[random_tid].opinion_local = sum1
        # print(random_tid,"Is talking to his freinds ")

        for tid in trader_list:
            trader_list[tid].opinion_global = delta_p_norm/2  # this is the value of the input b
            trader_list[tid].opinion_event = event_opinion/2

            # todo: identity fn may not be best choice here -- may need to run it through a nonlinear fn e.g. logistic?
            # I don not think we need this
            # update overall opinion to reflect new global opinion

            rate = (-trader_list[tid].obstinant * trader_list[tid].opinion) + (trader_list[tid].attention * (
                math.tanh(trader_list[tid].opinion + trader_list[tid].opinion_local))) + \
                   (trader_list[tid].opinion_global + trader_list[tid].opinion_event)

            trader_list[tid].opinion = max(min((trader_list[tid].opinion + (timestep * rate)), 1), -1)

    def socnet_connect(trader_list):
        # connect each trader's social network -- for opinion dynamics
        # and give it an initial opinion too
        # current version: social network is fully connected graph, all strengths are random
        # something to explore: what if sellers only connect to sellers, and buyers to buyers?
        print("socnet_connect is called")

        inital_weight = 1
        G = nx.MultiDiGraph()

        # add the nodes
        G.add_nodes_from(trader_list)

        for node in trader_list:
            for neighbor_node in trader_list:
                if node != neighbor_node:
                    G.add_edges_from([(node, neighbor_node)], weight=inital_weight)

                """if 'B' in node and neighbor_node and node != neighbor_node:
                    G.add_edges_from([(node, neighbor_node)], weight= -2)"""

        """ just for viusalization
        pos = nx.spring_layout(G)
        nx.draw(G, pos)

        edge_labels = dict([((u, v,), d['weight'])
                            for u, v, d in G.edges(data=True)])

        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7)
        #plt.show()"""

        return G
        # end of socnet_connect

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False
    opinion_verbose = False
    strat_dump = open(sess_id + '_strats.csv', 'w')  # one line is writin in each iteration
    Opinion_dump = open(sess_id + '_opinions.csv', 'w')
    transtion_dump = open(sess_id +'_transactions.csv','w')
    # initialise the exchange

    exchange = Exchange()

    # create a bunch of traders
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, populate_verbose)
    for tid in traders.keys():
        traders[tid].opinion_t0 = random.uniform(-1, 1)
        traders[tid].opinion = traders[tid].opinion_t0
        #print(tid, traders[tid].opinion_t0, traders[tid].opinion)

    """ Social Netwrok creating and affinity/ adjacency matrix"""
    SN = socnet_connect(traders)
    """ we need to get the affinity/ adjacency matrix of the graph"""
    A = nx.to_numpy_matrix(SN, nodelist=None, dtype=None, order='F', nonedge=0.0)
    A = pd.DataFrame(A, index=[trader for trader in traders], columns=[trader for trader in traders])

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    duration = float(endtime - starttime)

    last_update = -1.0

    time = starttime

    pending_cust_orders = []

    if verbose:
        print('\n%s;  ' % sess_id)

    # frames_done is record of what frames we have printed data for thus far
    frames_done = set()
    trans_recording = []

    print("time starts at ", time, "time ends at ", endtime)

    while time < endtime:

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration

        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

        trade = None

        [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)
        # for order in  pending_cust_orders:
        # print(order)
        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].lastquote, verbose)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lob_verbose))

        # print("this ans order to trader", order)
        # if verbose: print('Trader Quote: %s' % (order))

        if order is not None:
            if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order2(time, order, process_verbose)

            if trade is not None:
                trans_recording.append(trade)
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                if dump_all:
                    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))

            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lob_verbose)



            opinion_dynamics(time, timestep, lob, traders, A, trans_recording, opinion_verbose)
            """
            for tid in traders.keys():
                if time_left <=0.4:
                    traders[tid].opinion =+1.0
                elif time_left <=0.7:
                    traders[tid].opinion = 0.0
            
            for tid in traders.keys():

                if 'B' in tid:
                    traders[tid].opinion = -1.0
                if 'S'in tid:
                    traders[tid].opinion = -1.0
            """

            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                traders[t].respond(time, lob, trade, respond_verbose)

            # log all the PRSH/PRD/etc strategy info for this timestep?
            frame_rate = 60 * 60  # print one frame every this many simulated seconds

            if int(time) % frame_rate == 0 and int(time) not in frames_done:
                # print one more frame to strategy dumpfile
                dump_strats_frame(time, strat_dump, Opinion_dump, traders)

                exchange.tape_dump(transtion_dump, 'keep')
                # record that we've written this frame
                frames_done.add(int(time))

        time = time + timestep
    print("we have DONE with sesson at time", time)
    # session has ended

    strat_dump.close()
    Opinion_dump.close()
    transtion_dump.close()
    if dump_all:

        # dump the tape (transactions only -- not dumping cancellations)
        #exchange.tape_dump(sess_id + '_transactions.csv', 'w', 'keep')

        # record the blotter for each trader
        bdump = open(sess_id + '_blotters.csv', 'w')
        for t in traders:
            bdump.write('%s, %d\n' % (traders[t].tid, len(traders[t].blotter)))
            for b in traders[t].blotter:
                bdump.write('%s, Blotteritem, %s\n' % (traders[t].tid, b))
        bdump.close()

    # write trade_stats for this session (NB end-of-session summary only)
    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))


#############################

# # Below here is where we set up and run a series of experiments

import timeit

if __name__ == "__main__":
    t1 = timeit

    # set up common parameters for all market sessions
    n_days = 50  # 1000 days is good, but 3*365=1095, might as well go for three years.
    start_time = 0.0
    end_time = 60.0 * 60.0 * 24 * n_days
    duration = end_time - start_time


    # schedule_offsetfn returns time-dependent offset, to be added to schedule prices
    def schedule_offsetfn(t):

        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 100 * t / (c / pi2)
        amplitude = 100 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))


    # Here is an example of how to use the offset function
    #
    # range1 = (10, 190, schedule_offsetfn)
    # range2 = (200,300, schedule_offsetfn)

    # Here is an example of how to switch from range1 to range2 and then back to range1,
    # introducing two "market shocks"
    # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.

    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #               {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #             {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #        ]
    # demand_schedule = supply_schedule
    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100

    # range1 = (60, 60)
    # supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

    # range2 = (100, 100)
    # demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

    # The code below sets up flat (perfectly elastic) supply and demand curves at prices of 50 and 150, P0=100

    range1 = (60, 60)
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

    range2 = (140, 140)
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

    # new customer orders arrive at each trader approx once every order_interval seconds
    order_interval = 5

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': order_interval, 'timemode': 'drip-poisson'}

    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'order_interval': 30, 'timemode': 'periodic'}

    # buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
    # sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]

    opponent = 'GVWY'
    opp_N = 30
    #    sellers_spec = [('PRSH', 30),(opponent, opp_N-1)]
    #    buyers_spec = [(opponent, opp_N)]

    # sellers_spec = [('PRSH',6, {'k': 4, 's_min': -1.0, 's_max': +1.0}),
    # ('PRDE',6, {'k': 4, 's_min': -1.0, 's_max': +1.0})]

    sellers_spec = [('OPRDE', 30, {'k': 4, 's_min': -1.0, 's_max': +1.0})]
    buyers_spec = [('OPRDE', 30, {'k': 4, 's_min': -1.0, 's_max': +1.0})]
    #sellers_spec = [('PRZI', 30)]
    #buyers_spec = [('PRZI', 30)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # run a sequence of trials, one session per trial

    verbose = False

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 10

    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 10

    trial = 1

    while trial < (n_trials + 1):
        # create unique i.d. string for this trial
        trial_id = 'bse_OPRDE_OD_d%03d_i%02d_%04d' % (n_days, order_interval, trial)

        if trial > n_trials_recorded:
            dump_all = False
        else:
            dump_all = True

        dump_all = True
        tdump = open(trial_id + 'avg_balance.csv', 'w')

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

        # cProfile.run("market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)",filename="out.prof")
        # market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

        tdump.close()
        print("we have DONE with Trial number", trial)
        trial = trial + 1

    print(t1.timeit())

    # run a sequence of trials that exhaustively varies the ratio of four trader types
    # NB this has weakness of symmetric proportions on buyers/sellers -- combinatorics of varying that are quite nasty
    #
    # n_trader_types = 4
    # equal_ratio_n = 4
    # n_trials_per_ratio = 50
    #
    # n_traders = n_trader_types * equal_ratio_n
    #
    # fname = 'balances_%03d.csv' % equal_ratio_n
    #
    # tdump = open(fname, 'w')
    #
    # min_n = 1
    #
    # trialnumber = 1
    # trdr_1_n = min_n
    # while trdr_1_n <= n_traders:
    #     trdr_2_n = min_n
    #     while trdr_2_n <= n_traders - trdr_1_n:
    #         trdr_3_n = min_n
    #         while trdr_3_n <= n_traders - (trdr_1_n + trdr_2_n):
    #             trdr_4_n = n_traders - (trdr_1_n + trdr_2_n + trdr_3_n)
    #             if trdr_4_n >= min_n:
    #                 buyers_spec = [('GVWY', trdr_1_n), ('SHVR', trdr_2_n),
    #                                ('ZIC', trdr_3_n), ('ZIP', trdr_4_n)]
    #                 sellers_spec = buyers_spec
    #                 traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}
    #                 # print buyers_spec
    #                 trial = 1
    #                 while trial <= n_trials_per_ratio:
    #                     trial_id = 'trial%07d' % trialnumber
    #                     market_session(trial_id, start_time, end_time, traders_spec,
    #                                    order_sched, tdump, False, True)
    #                     tdump.flush()
    #                     trial = trial + 1
    #                     trialnumber = trialnumber + 1
    #             trdr_3_n += 1
    #         trdr_2_n += 1
    #     trdr_1_n += 1
    # tdump.close()
    #
    # print(trialnumber)

