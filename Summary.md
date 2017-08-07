# Chapter 1

##### Main components of credit risk
$$ E[loss]=PD \times LGD \times EAD$$

- __Probability of default__ 
  > Mitigate with __Guarantee__ or __CDS__
- __Exposure at default__ ( Potential future exposure , notional value)
  > mitigate with __Netting__
- loss Given Default 
  > Mitigate with __collateral__

##### Other stuffs
- Procyclicality 
  > PD and LGD move together because of business cycle
- Wrong way risk
  > PD and EAD move together. When people know they will default they draw on their credit 

- LGD can be >1
Collection process also costs money and human capital 

- LGD <0 Penalty in case of default + you have access to taxation. Municipalities like the Lannisters always pay their debts 
- S&P rating 
  >AAA - AA - A - __BBB__(investment grade) - __BB__(speculative)- B - CCC - D
- __Credit Scoring__
Through the cycle( what we did) versus Point in Time
  >  Difficult to backtest credit risk 

- Migration Matrix Regularization
Need to do shit to it to make it better behaved 

# Chapter 2
You know this shit better than me 
# Chapter 3
Estimating PD
1. Historical data
2. Credit Ratings
3. CDS 
4. Merton-like models

#### Merton model
If Assets < Liabilities --> default
Assets = underlying latent variable
Liabilities = default barrier 
Resort to option pricing --> Equity is a European call option on the Assets
$$Pr(Default) = P(A(t)<DB) = 1-N(d2) = N(-d2)$$

__Flaws of the model__
> - Can not observe Asset volatility(deduce from equity volatility)
> - Can not observe Asset Prices
> - It's not a European option but a __Barrier__ option
> - The world is not Gaussian
> - We should be using real world measure not risk neutral measure
##### KMV Moody's = Extension Maturity
- Moody's works with the notion of distance to default = how many standard deviations am I from default
- $N( -d2)$ is estimated from proprietary data set 
- Takes into account first passage idea of Barrier option
- Default barrier is estimated more accurately(short term liabilities blabla)
##### Inferring PD from market : Credit Default Swap
__Definition__ : Contract where company A has the right to sell a bond issued by company C for its face value to company B in the event of default
- Fix Recovery rate at 40% = market convention Goldman Sachs
- Protection Buyer Versus Protection Seller
- Spread = premium = Bond yield - risk free Bond
- Getting Default term structure : too many payments for too little CDS quotes.  

__Solution__
1. Bootstrapping --> possible to get negative Spread
2. The Martin-Thompson Approach solves this
# Chapter 4 
1. Expected Loss = Cost of doing Business 
2. Unexpected Loss
> - Risk Capital
>- Economic Capital

Basel Capital requirements refer to unexpected losses 

- __Copula__ a construct to link the independent marginal distributions to a joint distribution
- __Attributes Risk Measure__ :
VAR violates in some cases sub-additivity(especially for non granular portfolios ):
$$VAR(a+b)> Var(a)+ Var(b)) $$
Don't have this problem for Expected Shortfall
- __Correlation Modelling__
Define Correlation through factors :
> 1. Basel Formula has 1 factor 
> 2. __CreditMetrics__ : 2 factors Country, Industry  --> Calibrate on industry Country indices e.g CAC40 , FTSE100

#### Basel III formula 
- One factor model where number of obligors goes to infinity
- Analytic Solution
- Exact contribution to Value at Risk 
- __Conditional independence__ : Given that I know state of factor variable, I can move conditional expectation of loss distribution inside value at risk operator
- __Maturity Adjustment__ To take into account rating changes, biug

__Problems with Basel Formula__ 
 assumes large protfolios with small exposures. Breaks down if you have one big counterparty , creates a kink in loss distribution
 
# Chapter 5 Credit Risk +
lots of Math --> Poisson distribution
Exact analytic solution 


# Chapter 6 
__Stress Testing__

1. Sensitivity analysis Bottom up = Move market variables : yield curve , spot volatilities, rating changes
2. Event driven Scenario analysis = Top down --> look at big event


