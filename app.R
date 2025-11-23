# Ensure missing dependencies install on Connect
if (!requireNamespace("litedown", quietly = TRUE)) {
    install.packages("litedown")
}

# ---- Packages ----
# Shiny
library(shiny)

# Core tidyverse pieces (no meta-package)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(purrr)
library(stringr)
library(forcats)
library(tibble)

# Time series + forecasting
library(tsibble)
library(fable)
library(fabletools)
library(feasts)
library(lubridate)

# ---- Data: load & wrangle ----
# Make sure the file in your project is exactly "AustralianWines.csv"
wines_raw <- read_csv("AustralianWines.csv")

# Clean column names
names(wines_raw) <- trimws(names(wines_raw))

# Convert all varietal columns to numeric (Rose had chars)
wines_raw <- wines_raw |>
    mutate(across(-Month, ~ as.numeric(.)))

# Wide → long: Month, Varietal, Sales
wines_long <- wines_raw |>
    pivot_longer(
        cols = -Month,
        names_to = "Varietal",
        values_to = "Sales"
    )

# Parse Month like "Jan-80" into Date and make a tsibble
wines_ts <- wines_long |>
    mutate(
        Month = my(Month)   # lubridate::my("Jan-80") -> 1980-01-01
    ) |>
    filter(!is.na(Month)) |>
    as_tsibble(index = Month, key = Varietal)

min_month <- min(wines_ts$Month)
max_month <- max(wines_ts$Month)
default_train_end <- max_month %m-% years(3)

# ---- UI ----
ui <- fluidPage(
    titlePanel("Australian Wine Sales Forecast Explorer"),
    
    sidebarLayout(
        sidebarPanel(
            selectInput(
                "varietal", "Select varietal(s):",
                choices = sort(unique(wines_ts$Varietal)),
                selected = "Red",
                multiple = TRUE
            ),
            
            sliderInput(
                "date_range", "Date range:",
                min = min_month,
                max = max_month,
                value = c(min_month, max_month),
                timeFormat = "%Y-%m"
            ),
            
            sliderInput(
                "train_end", "Training end date:",
                min = min_month,
                max = max_month,
                value = default_train_end,
                timeFormat = "%Y-%m"
            ),
            
            numericInput(
                "h", "Forecast horizon (months):",
                value = 12, min = 1, max = 60, step = 1
            ),
            
            helpText(
                "Choose varietals and a date range, then set a training end date ",
                "to define the validation window and forecast horizon."
            )
        ),
        
        mainPanel(
            tabsetPanel(
                tabPanel(
                    "Overview",
                    h4("Filtered Data Preview"),
                    tableOutput("preview"),
                    br(),
                    h4("Overview Plot"),
                    plotOutput("overview_plot"),
                    br(),
                    h4("Seasonal Plot"),
                    plotOutput("season_plot")
                ),
                tabPanel(
                    "Forecasts & Models",
                    h4("Forecast Error Summary"),
                    verbatimTextOutput("fc_summary"),
                    br(),
                    h4("Comparative Forecasts"),
                    plotOutput("forecast_plot"),
                    br(),
                    h4("Accuracy (Training vs Validation)"),
                    tableOutput("accuracy_table"),
                    br(),
                    h4("Model Specifications"),
                    tableOutput("specs_table")
                )
            )
        )
    )
)

# ---- Server ----
server <- function(input, output, session) {
    
    # Keep training end inside selected date range
    observe({
        dr <- input$date_range
        if (input$train_end < dr[1] || input$train_end > dr[2]) {
            updateSliderInput(session, "train_end", value = dr[2])
        }
    })
    
    # Filtered data (tsibble, no NAs in Sales)
    filtered_data <- reactive({
        wines_ts |>
            filter(
                Varietal %in% input$varietal,
                Month >= input$date_range[1],
                Month <= input$date_range[2]
            ) |>
            filter(!is.na(Sales))
    })
    
    # ---------- OVERVIEW TAB ----------
    
    output$preview <- renderTable({
        dat <- filtered_data()
        if (nrow(dat) == 0) return(NULL)
        head(dat, 10)
    })
    
    output$overview_plot <- renderPlot({
        dat <- filtered_data()
        if (nrow(dat) == 0) return(NULL)
        
        dat |>
            ggplot(aes(Month, Sales, color = Varietal)) +
            geom_line() +
            theme_minimal() +
            labs(
                title = "Overview of Selected Wine Sales",
                x = "Month",
                y = "Sales"
            )
    })
    
    # Custom seasonal plot (months on x, coloured by year)
    output$season_plot <- renderPlot({
        dat <- filtered_data()
        if (nrow(dat) == 0) return(NULL)
        
        dat |>
            mutate(
                Year        = year(Month),
                MonthInYear = month(Month, label = TRUE, abbr = TRUE)
            ) |>
            ggplot(aes(MonthInYear, Sales, group = Year, color = factor(Year))) +
            geom_line(alpha = 0.7) +
            facet_wrap(vars(Varietal), scales = "free_y") +
            theme_minimal() +
            labs(
                title = "Seasonal Pattern by Varietal",
                x = "Month of Year",
                y = "Sales",
                color = "Year"
            )
    })
    
    # ---------- TRAIN / VALIDATION SPLIT (NO filter_index) ----------
    
    train_data <- reactive({
        dat <- filtered_data()
        if (nrow(dat) == 0) return(dat)
        dat |>
            filter(Month <= input$train_end)
    })
    
    valid_data <- reactive({
        dat <- filtered_data()
        if (nrow(dat) == 0) return(dat)
        dat |>
            filter(Month > input$train_end)
    })
    
    # ---------- MODELS (TSLM, ETS, ARIMA) ----------
    
    models <- reactive({
        dat_trn <- train_data()
        if (nrow(dat_trn) < 24) return(NULL)  # need at least 2 years of monthly data
        
        dat_trn |>
            model(
                TSLM  = TSLM(Sales ~ trend() + season()),
                ETS   = ETS(Sales),
                ARIMA = ARIMA(Sales)
            )
    })
    
    fc_future <- reactive({
        mdl <- models()
        if (is.null(mdl)) return(NULL)
        mdl |> forecast(h = input$h)
    })
    
    fc_valid <- reactive({
        mdl <- models()
        vd  <- valid_data()
        if (is.null(mdl) || nrow(vd) == 0) return(NULL)
        mdl |> forecast(new_data = vd)
    })
    
    # ---------- ACCURACY TABLE ----------
    
    accuracy_table <- reactive({
        mdl <- models()
        if (is.null(mdl)) return(tibble())
        
        acc_train <- mdl |>
            accuracy() |>
            mutate(Set = "Training")
        
        vd <- valid_data()
        acc_all <- acc_train
        
        fc_v <- fc_valid()
        if (!is.null(fc_v) && nrow(vd) > 0) {
            acc_valid <- fc_v |>
                accuracy(vd) |>
                mutate(Set = "Validation")
            acc_all <- bind_rows(acc_train, acc_valid)
        }
        
        acc_all |>
            select(Varietal, .model, Set, RMSE, MAE, MAPE) |>
            arrange(Varietal, .model, Set)
    })
    
    output$accuracy_table <- renderTable({
        acc <- accuracy_table()
        if (nrow(acc) == 0) return(NULL)
        acc
    }, digits = 3)
    
    # ---------- MODEL SPECIFICATIONS ----------
    
    specs_table <- reactive({
        mdl <- models()
        if (is.null(mdl)) return(tibble())
        
        g <- mdl |> glance()
        
        # Ensure columns exist for all models (TSLM won't have them)
        needed_chr <- c("error", "trend", "season")
        for (nm in needed_chr) {
            if (!nm %in% names(g)) g[[nm]] <- NA_character_
        }
        
        needed_int <- c("p", "d", "q", "P", "D", "Q")
        for (nm in needed_int) {
            if (!nm %in% names(g)) g[[nm]] <- NA_integer_
        }
        
        g |>
            mutate(
                ETS_form = if_else(
                    .model == "ETS",
                    paste0("ETS(", error, ",", trend, ",", season, ")"),
                    NA_character_
                ),
                ARIMA_order = if_else(
                    .model == "ARIMA",
                    paste0("ARIMA(",
                           p, ",", d, ",", q, ")(",
                           P, ",", D, ",", Q, ")[12]"),
                    NA_character_
                )
            ) |>
            select(Varietal, .model, ETS_form, ARIMA_order, AIC, AICc, BIC) |>
            arrange(Varietal, .model)
    })
    
    output$specs_table <- renderTable({
        st <- specs_table()
        if (nrow(st) == 0) return(NULL)
        st
    }, na = "")
    
    # ---------- FORECAST ERROR SUMMARY (EXTRA FEATURE) ----------
    
    best_model_summary <- reactive({
        acc <- accuracy_table()
        if (nrow(acc) == 0)
            return("Not enough data in the selected window to compare models.")
        
        acc_val <- acc |> filter(Set == "Validation")
        acc_use <- if (nrow(acc_val) > 0) acc_val else acc |> filter(Set == "Training")
        
        best <- acc_use |>
            group_by(Varietal) |>
            slice_min(RMSE, n = 1, with_ties = FALSE) |>
            ungroup() |>
            mutate(
                line = paste0(
                    "For ", Varietal,
                    ", the best model is ", .model,
                    " (", Set, " set) with RMSE = ", round(RMSE, 2),
                    ", MAE = ", round(MAE, 2),
                    ", MAPE = ", round(MAPE, 2), "%."
                )
            )
        
        paste(best$line, collapse = "\n")
    })
    
    output$fc_summary <- renderText({
        best_model_summary()
    })
    
    # ---------- FORECAST PLOT ----------
    
    output$forecast_plot <- renderPlot({
        dat <- filtered_data()
        fc  <- fc_future()
        if (is.null(fc) || nrow(dat) == 0) return(NULL)
        
        dat |>
            autoplot(Sales) +
            autolayer(fc, alpha = 0.6) +
            facet_wrap(vars(Varietal), scales = "free_y") +
            theme_minimal() +
            labs(
                title = "Comparative Forecasts: TSLM vs ETS vs ARIMA",
                x = "Month",
                y = "Sales"
            )
    })
}

shinyApp(ui = ui, server = server)
