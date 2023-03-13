library(circlize)
library(corrplot)
library(lattice)

# see
# https://stackoverflow.com/questions/68080870/chord-plot-for-a-correlation-matrix-r
# https://stackoverflow.com/questions/31943102/rotate-labels-in-a-chorddiagram-r-circlize

db <- read.csv(file = '/root/cardiac/Ageing/marco/original_db.csv')

db2 <- db[,-c(grep("long_", names(db)), grep("rad_", names(db)))]
db2$rad_2 <- db$rad_2
db2$long_2 <- db$long_2
names(db2)[names(db2) %in% c("AAo_maxarea")] = "AAo_max"
names(db2)[names(db2) %in% c("AAo_minarea")] = "AAo_min"
names(db2)[names(db2) %in% c("Aao_dist")] = "Aao_dis"
names(db2)[names(db2) %in% c("DAo_maxarea")] = "DAo_max"
names(db2)[names(db2) %in% c("DAo_minarea")] = "DAo_min"
names(db2)[names(db2) %in% c("Dao_dist")] = "DAo_dis"
names(db2)[names(db2) %in% c("corr_IVS")] = "IVS"

mat <- cor(db2)
mat[lower.tri(mat, TRUE)] <- NA
 
df = data.frame(from = rep(rownames(mat), times = ncol(mat)),
                to = rep(colnames(mat), each = nrow(mat)),
                value = as.vector(mat),
                stringsAsFactors = FALSE)
df <- df[!is.na(df[,3]),]
df[,3] <- abs(df[,3])
df <- df[sort(df[,3], index.return=TRUE)$ix,]

#df <- df[df[,1]!=df[,2],]
df = df[df[,3]>0.4,]
cols = colorRamp2(c(.6,.8,.85, .95,1),c(
"#fbfbfb",
"#c6e3bc", 
"#3c5e3b", 
"#002400",
"black"
),
transparency = .1)


pdf(file = "/root/cardiac/Ageing/marco/circos_plot.pdf")
#chordDiagram(df, col=cols, big.gap=30, annotationTrack =  c("name", "grid"))

chordDiagram(df,col=cols, annotationTrack = "grid", preAllocateTracks = 1)
circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  circos.text(mean(xlim), ylim[1] + .1, sector.name, facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5))
  circos.axis(h = "top", labels.cex = 0.5, major.tick.percentage = 0.2, sector.index = sector.name, track.index = 2)
}, bg.border = NA)


dev.off()


db3 <- db[,-c(grep("long_", names(db)), grep("rad_", names(db)))]
pca = prcomp(db[,grep('long_', names(db))], center = TRUE, scale. = TRUE)
db3$long_pc_1 <- pca$x[,1]
db3$long_pc_2 <- pca$x[,2]

pca = prcomp(db[,grep('rad_', names(db))], center = TRUE, scale. = TRUE)
db3$rad_pc_1 <- pca$x[,1]
db3$rad_pc_2 <- pca$x[,2]

print(names(db3) == c("LVEDV", "LVESV", "LVSV", "LVEF", "LVCO",
"LVM", "sex", "AAo_maxarea", "AAo_minarea", "Aao_dist",
"DAo_maxarea", "DAo_minarea", "Dao_dist", "RVEDV", "RVESV",
"RVSV", "RVEF", "RAV_min", "RASV", "RAEF",
"RAV_max", "LAV_max", "LAV_min", "LASV", "LAEF",
"corr_IVS", "long_pc_1", "long_pc_2", "rad_pc_1", "rad_pc_2"))

names(db3) <- c("LVEDV", "LVESV", "LVSV", "LVEF", "LVCO",
"LVM", "sex", "Asc aorta max. area", "Asc aorta min. area", "Asc aorta dist",
 "Desc aorta max. area", "Desc aorta min. area", "Desc aorta dist", "RVEDV", "RVESV",
"RVSV", "RVEF", "RA min. vol", "RASV", "RAEF",
"R atrium max. volume", "L atrial max. vol", "L atrial min. vol", "LASV", "LAEF",
"T1 (septum)", "Long SR PC1", "Long SR PC2", "Radial SR PC1", "Radial SR PC2")
write.csv(db3, '/root/cardiac/Ageing/marco/original_db_with_principal_components.csv', row.names = FALSE)

pdf(file = "/root/cardiac/Ageing/marco/heatmap.pdf")
superheat(cor(db3), 
row.dendrogram = T, 
col.dendrogram = T,
left.label.text.size = 2,
bottom.label.text.size = 2,
left.label.size = 0.22,
bottom.label.size = 0.26,
# rotate bottom label text
bottom.label.text.angle = 90,
heat.col.scheme = "red",
grid.hline.col = "white",
grid.vline.col = "white",
grid.hline.size = .7,
grid.vline.size = .7,
#legend.breaks = c(-1., 0, 1)
)
dev.off()


pdf(file = "/root/cardiac/Ageing/marco/heatmap2.pdf")
heatcolor = heat.colors(10000, rev=TRUE)
heatcolor[10000] = '#7F0000'
levelplot(cor(db3), col.regions=heatcolor, scales=list(y=list(rot=0), x=list(rot=70)), xlab="", ylab="")
dev.off()



pdf(file = "/root/cardiac/Ageing/marco/corr_plot.pdf")
corrplot(cor(db3))
dev.off()

