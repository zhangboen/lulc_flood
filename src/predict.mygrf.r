#object = local Forests
library(proxy)
predict.grf2 <- function(object, new.data, x.var.name, y.var.name, Attrs, geoW = 0.5, attW = 0.5, local.w=1, global.w=0, ...) {

  Obs <- nrow(new.data)

  predictions <- vector(mode="numeric", length=Obs)
  predictions.Global <- vector(mode="numeric", length=Obs)

  # get geographic locations and attributes
  loc.new <- new.data[c(x.var.name,y.var.name)]
  loc.old <- object$Locations
  attrs.new <- new.data[Attrs]
  attrs.old <- object$TAttrs

  # get the nearest model ID for each row in new.data
  if (Obs < 10000) {
    # calculate geographic distance
    D <- proxy::dist(loc.new, loc.old, method="Euclidean")

    # calculate attribute distance
    ATTR <- proxy::dist(attrs.new, attrs.old, method="Euclidean")

    # combine geographic and attribute distance
    D <- D * geoW + ATTR * attW

    # get the nearest model ID
    model.id <- apply(D, 1, which.min)
  } else {
    model.id <- vector(mode="numeric", length=Obs)
    s <- 1:Obs
    idx <- split(s, ceiling(seq_along(s)/5000))
    for (i in 1:length(idx)) {
      chunk <- idx[[i]]
      loc.new0 <- loc.new[chunk,]
      attrs.new0 <- attrs.new[chunk,]
      # calculate geographic distance
      D <- proxy::dist(loc.new0, loc.old, method="Euclidean")

      # calculate attribute distance
      ATTR <- proxy::dist(attrs.new0, attrs.old, method="Euclidean")

      # combine geographic and attribute distance
      D <- D * geoW + ATTR * attW

      # get the nearest model ID
      model.id0 <- apply(D, 1, which.min)
      model.id[chunk] <- model.id0
      print(paste('calculate combined distance', i))
    }
  }

  # loop for each sub-model
  model.id.unique <- unique(model.id)
  for (model.id0 in model.id.unique) {
    rows <- which(model.id == model.id0)
    new.data0 <- new.data[rows,]
    local.model.ID <- model.id0
    
    g.predict <- predict(object[[1]], new.data0, ...)
    g.prediction <- g.predict$predictions
    l.predict <- predict(object$Forests[[local.model.ID]], new.data0)
    l.prediction <- l.predict$predictions

    predictions[rows] <- global.w * g.prediction + local.w * l.prediction
    predictions.Global[rows] <- g.prediction
    print(paste('done', model.id0))
  }
  # for(i in 1:Obs){

  #   x <- new.data[i, which(names(new.data)==x.var.name)]
  #   y <- new.data[i, which(names(new.data)==y.var.name)]
  #   attrs <- new.data[i, Attrs]

  #   locations <- object$Locations
  #   TAttrs <- object$TAttrs

  #   D <- sqrt((x-locations[,1])^2 + (y-locations[,2])^2) / 1000 
  #   ATTR <- sqrt(apply(abs(mapply('-', TAttrs, attrs))^2, 1, sum))
  #   D <- D * geoW + ATTR * attW

  #   local.model.ID <- which.min(D)

  #   g.predict <- predict(object[[1]], new.data[i,], ...)
  #   g.prediction <- g.predict$predictions
  #   l.predict <- predict(object$Forests[[local.model.ID]], new.data[i,])
  #   l.prediction <- l.predict$predictions

  #   predictions[i] <- global.w * g.prediction[1] + local.w * l.prediction[1]
  #   if (i%%500 == 0)
  #     print(paste('done', i))
  # }
return(list(predictions.L = predictions, predictions.G = predictions.Global))
}