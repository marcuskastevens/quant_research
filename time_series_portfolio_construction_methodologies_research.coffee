
"""

General Model Transforms:

    E[r] Transforms:
        Asset-Level E[r]:
            Reverse MVO
            Cross-model mean
            Cross-model weighted mean
            Discount the prior two methodologies
            Cross-model binary voting system: potentially (m / N) / vol
            ML        
        Model-Level E[r]:
            Aggregate weighted reverse MVO
            Aggregated weighted cross-model mean
            Aggregated weighted cross-model weighted mean
            Bayesian update of model-level E[r]
            ML
    Weight Transforms:
        Asset-Level:
            Equal risk weight
            Equal vol weight with portfolio vol scalar
            Trailing z-score / sum(trailing z-score) --  perhaps trunctated attractiveness score at 2std -- scale each z-score by inverse vol
            Asset-level max sharpe opt + kelly bet of SR / vol
            Discrete kelly bet based on MC sim and stop-loss (simulate probability of loss vs. implied odds... adjust implied odds accordingly)
            Alternatie opts (max probabilistic portfolio, min VaR, max sortino, etc.)
        Model-Level:
            Model-level max sharpe opt + kelly bet of SR / vol
            Equal risk weight
            Equal vol weight with portfolio vol scalar
            Alternatie opts (max probabilistic portfolio, min VaR, max sortino, etc.)


        
            



            

TS Models:

    Portfolio 0:
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            equal risk weight signals across universe
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        compute equal risk model weights
        apply kelly bet of SR / ex-ante vol

    Portfolio 1:    
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            equal risk weight asset-level signals across universe
            compute implied E[r] via reverse MVO        
        compute E[r] for each asset via cross-model mean (or potentially model-weighted mean based on model conviction) 
        compute vector of E[r] significance-based discount scalars = sqrt(n) / max(sqrt(n for n estimates for each asset)) 
            # this will discount every E[r] not associated with the E[r] with the most estimates
        compute weighted E[r] by scaling each E[r] by the corresponding discount rate from the previously computed vector
        run max SR optimization
        apply kelly bet of SR / ex-ante vol
    
    Portfolio 2:
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            equal risk weight asset-level signals across universe
            compute implied expected returns via reverse MVO        
        compute E[r] for each asset via cross-model mean (or potentially model-weighted mean based on model conviction) 
        compute vector of E[r] significance-based discount scalars = sqrt(n) / max(sqrt(n for n estimates for each asset)) 
            # this will discount every E[r] not associated with the E[r] with the most estimates
        compute weighted E[r] by scaling each E[r] by the corresponding discount rate from the previously computed vector
        compute each model's aggregate weighted E[r] based on the algorithm above
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        run max SR optimization
        apply kelly bet of SR / ex-ante vol
    
    Portfolio 3:
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            equal risk weight asset-level signals across universe
            compute implied expected returns via reverse MVO    
            compute aggregate E[r] of model
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        run max SR optimization
        apply kelly bet of SR / ex-ante vol

    Portfolio 4:
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            equal risk weight asset-level signals across universe
            compute implied expected returns via reverse MVO    
            compute aggregate E[r] of model
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        compute equal risk model weights
        apply kelly bet of SR / ex-ante vol
    
    Portfolio 5:
        for each model:
            generate binary/indicator buy/flat/sell signal (-1, 0, 1)
            inverse vol weight asset-level signals across universe
            apply rolling vol scalar to target vol
        inverse vol weight models
        apply rolling model vol scalar to target vol
        

    Portfolio 6:
        for each model:
            generate trailing z-score of each asset's signal's magnitude
            weight across each asset's z-score (z-score / sum(z-score) for each asset's z-score)
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        compute equal risk model weights
        apply kelly bet of SR / ex-ante vol

    Portfolio 7:
        for each model:
            generate trailing z-score of each asset's signal's magnitude
            weight across each asset's z-score (z-score / sum(z-score) for each asset's z-score)
            compute model's aggregate weighted implied E[r] from reverse MVO 
        compute each model's ex-ante risk
        compute corr matrix of each model (based on correlation of weight vectors / backtest)
        run max SR optimization
        apply kelly bet of SR / ex-ante vol
    
    Portfolio 8:
        for each model:
            generate trailing z-score of each asset's signal's magnitude
            weight across each asset's z-score (z-score / sum(z-score) for each asset's z-score)
            compute implied expected returns via reverse MVO        
        compute E[r] for each asset via cross-model mean (or potentially model-weighted mean based on model conviction) 
        compute vector of E[r] significance-based discount scalars = sqrt(n) / max(sqrt(n for n estimates for each asset)) 
            # this will discount every E[r] not associated with the E[r] with the most estimates
        compute weighted E[r] by scaling each E[r] by the corresponding discount rate from the previously computed vector
        run max SR optimization
        apply kelly bet of SR / ex-ante vol


        

    





    
equal risk weight models across portfolio (compute each ex-ante risk from the underlying equal risk weighted models)

    apply ensemble of expected returns methodologies:
        compute implied expected returns from reverse MVO
        compute expected returns from ML models
        compute expected returns from non-parametric cross-model voting system to capture implied conviction



XS



""