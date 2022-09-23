# Contributing

Thanks for your interest! 
This is a very small package that isn't really meant to be standalone. 
So I suspect you are probably future me. :)
Otherwise, be wary that this is not really the best code.

First navigate to the `IRLQuestR` folder.
Then, install devtools [devtools](https://www.r-project.org/nosvn/pandoc/devtools.html)

```
install.packages("devtools")
```

## Updating documentation

If you don't already have roxygen2:

```
install.packages("roxygen2")
```

Then: 

```
devtools::document()
```

## Updating / running tests

If you don't already have testthat:

```
install.packages("testthat")
```

Then:

```
devtools::test()
```