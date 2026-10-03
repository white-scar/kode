#!/usr/bin/env Rscript
# Figure 7 (table2): per-trait PA + heritability, 6 panels (a_iter/a_rec/b/c/d/e)
# 逐行沿用 pic-code/fig7_maize.R，仅按数据集切换输入文件 / 表型表 / x 轴范围 / 输出名。
# 用法: Rscript fig7_table2.R maize|arab|rice
args <- commandArgs(trailingOnly = TRUE)
ds <- if (length(args) >= 1) args[1] else "maize"
CFG <- list(
  maize = list(label = "Maize MAGIC",
               pheno = "data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv",
               xr = c(15, 47)),
  arab  = list(label = "Arabidopsis",
               pheno = "data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv",
               xr = c(8, 21)),
  rice  = list(label = "Rice (indica, control)",
               pheno = "data/rice_salinity/rice_indica_control_traits.csv",
               xr = c(30, 42))
)[[ds]]
HERE <- file.path("reproduce/pic-code/E3-4", ds)
dir.create(HERE, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(reshape2))

title_size <- 8; axis_text_size <- 8

pa_tp <- read.csv(file.path(HERE, sprintf("figE8_%s_pa_bytp_v4.csv", ds)), stringsAsFactors = FALSE)
pa_tr <- read.csv(file.path(HERE, sprintf("figE8_%s_pa_bytrait_v4.csv", ds)), stringsAsFactors = FALSE)
herit <- read.csv(file.path(HERE, sprintf("h2_%s.csv", ds)), stringsAsFactors = FALSE)
colnames(herit)[colnames(herit) == "Heritability"] <- "H2"
trait_data <- read.csv(CFG$pheno, stringsAsFactors = FALSE)

# 5 representative traits from tp1+Meg+Dgp iter per-trait PA
# 2026-09-27: ??? PCC ??(???? max pcc >= 0.99)?"????"?
#   ?? max ?????????PCC~1 ?????(???? cog.y ????)
mx_pcc <- tapply(pa_tr$pcc, pa_tr$trait, max, na.rm = TRUE)
drop_tr <- names(mx_pcc)[mx_pcc >= 0.99]
cat(sprintf("dropped saturated traits (max pcc >= 0.99): %d\n", length(drop_tr)))
d_it_tr <- pa_tr[pa_tr$method == "tp1+Meg+Dgp_iter", ]
d_it_tr <- d_it_tr[!(d_it_tr$trait %in% drop_tr), ]
d_it_tr <- d_it_tr[order(d_it_tr$pcc), ]
qs <- c("min", "Q1", "mean", "Q3", "max")
n <- nrow(d_it_tr)
idxs <- round(c(1, n * 0.25, n * 0.5, n * 0.75, n))
rep5 <- setNames(d_it_tr$trait[idxs], qs)
cat("5 representative traits:\n"); print(rep5)
trait_colors <- c(max = "#00120B", Q3 = "#35605A", mean = "#6B818C", Q1 = "#709BFF", min = "#17DE6D")

make_compare_pa <- function(prot, title, ltype) {
  sub <- pa_tp[pa_tp$trait %in% rep5 & grepl(paste0(prot, "$"), pa_tp$method), ]
  sub$Method <- ifelse(grepl("tp1\\+knode", sub$method), "Kode+TP1", "DynamicGP+MegaLMM+TP1")
  sub$Method <- factor(sub$Method, levels = c("DynamicGP+MegaLMM+TP1", "Kode+TP1"))
  ggplot(sub, aes(x = tp, y = pcc, colour = Method, group = interaction(Method, trait))) +
    geom_line(linewidth = 0.5, linetype = ltype, alpha = 0.85) +
    scale_color_manual(values = c("DynamicGP+MegaLMM+TP1" = "#1F77B4", "Kode+TP1" = "#D62728")) +
    ylab("Prediction accuracy") + ggtitle(title) +
    scale_y_continuous(limits = c(-0.4, 1), breaks = seq(-0.4, 1, 0.2)) +
    scale_x_continuous(limits = CFG$xr) +
    theme_classic(base_family = "Times New Roman", base_size = 8) +
    theme(legend.position = "top", legend.box = "horizontal", legend.direction = "horizontal",
          legend.key.width = unit(0.25, "cm"), legend.key.height = unit(0.15, "cm"),
          text = element_text(family = "Times New Roman"),
          legend.title = element_text(size = 7, face = "bold"),
          legend.text = element_text(size = 7, face = "bold"),
          axis.title = element_text(size = title_size, face = "bold"),
          axis.text = element_text(size = axis_text_size, face = "bold"),
          title = element_text(size = 9, face = "bold")) +
    guides(colour = guide_legend(title = "Method", nrow = 1))
}
a_iter <- make_compare_pa("iter", "a_iter", "dashed")
a_rec  <- make_compare_pa("rec", "a_rec", "solid")

# b: representative-trait dynamics
traits_in <- intersect(rep5, colnames(trait_data))
td_sub <- trait_data[, c("bio_ID", "accession", "DAS", traits_in)]
days <- sort(unique(td_sub$DAS))
nd <- as.data.frame(matrix(NA, length(days), length(traits_in) + 1))
nd[, 1] <- days; colnames(nd) <- c("Day", traits_in)
row_of <- setNames(seq_along(days), as.character(days))
for (cat_ in traits_in) {
  inc <- which(colnames(td_sub) == cat_)
  for (day in days) {
    vals <- unlist(td_sub[td_sub$DAS == day, inc])
    nd[row_of[as.character(day)], cat_] <- mean(vals, na.rm = TRUE)
  }
}
nd[, -1] <- scale(nd[, -1])
nd_melt <- melt(nd, id.vars = 1); colnames(nd_melt) <- c("Day", "Trait", "Value")
nd_melt$Quantile <- NA
for (q in qs) nd_melt$Quantile[nd_melt$Trait == rep5[q]] <- q
nd_melt$Quantile <- factor(nd_melt$Quantile, levels = qs)
b <- ggplot(nd_melt, aes(x = Day, y = Value, colour = Quantile)) +
  geom_line(linewidth = 0.5) + ylab("Scaled trait value") + ggtitle("b") +
  scale_color_manual(values = trait_colors) +
  theme_classic(base_family = "Times New Roman", base_size = 8) +
  theme(text = element_text(family = "Times New Roman"),
        legend.position = c(0.3, 0.12), legend.key.width = unit(0.2, "cm"),
        legend.key.height = unit(0.1, "cm"), legend.title = element_text(size = 6, face = "bold"),
        legend.text = element_text(size = 6, face = "bold"),
        axis.title = element_text(size = title_size, face = "bold"),
        axis.text = element_text(size = axis_text_size, face = "bold"),
        title = element_text(size = 9, face = "bold")) +
  guides(colour = guide_legend(title = "Trait", ncol = 5))

# c: heritability
herit$DAS <- as.numeric(gsub("t", "", herit$Time))
h_abc <- herit[herit$Trait %in% rep5, ]
h_abc$Quantile <- NA
for (q in qs) h_abc$Quantile[h_abc$Trait == rep5[q]] <- q
h_abc$Quantile <- factor(h_abc$Quantile, levels = qs)
c_ <- ggplot(h_abc, aes(x = DAS, y = H2, colour = Quantile)) +
  geom_line(stat = "summary", linewidth = 0.5) +
  ylab("Heritability") + ggtitle("c") +
  scale_color_manual(values = trait_colors) +
  theme_classic(base_family = "Times New Roman", base_size = 8) +
  theme(legend.position = "none", text = element_text(family = "Times New Roman"),
        axis.title = element_text(size = title_size, face = "bold"),
        axis.text = element_text(size = axis_text_size, face = "bold"),
        title = element_text(size = 9, face = "bold"))

# rice: ? b/c ??? a ?????? x ?? (30-42)?DAS 28 ???
if (ds == "rice") {
  b  <- b  + scale_x_continuous(limits = CFG$xr)
  c_ <- c_ + scale_x_continuous(limits = CFG$xr)
}

# d/e: PA vs heritability CV
herit_tr <- aggregate(H2 ~ Trait, herit, FUN = mean)
herit_tr$SD <- aggregate(H2 ~ Trait, herit, FUN = sd)$H2
herit_tr$H.CV <- (herit_tr$SD / herit_tr$H2) * 100
herit_tr <- herit_tr[, c("Trait", "H.CV")]

make_cv <- function(pa_sub, title, dgp_col, kn_col) {
  pa_agg <- aggregate(pcc ~ trait + method, pa_sub, FUN = mean)
  tab_d <- merge(herit_tr, pa_agg[grepl("tp1\\+Meg\\+Dgp", pa_agg$method), c("trait", "pcc")], by.x = "Trait", by.y = "trait")
  tab_k <- merge(herit_tr, pa_agg[grepl("tp1\\+knode", pa_agg$method), c("trait", "pcc")], by.x = "Trait", by.y = "trait")
  tab_d$Source <- "DynamicGP+MegaLMM+TP1"; tab_k$Source <- "Kode+TP1"
  tab <- rbind(tab_d, tab_k)
  r_d <- cor(tab_d$H.CV, tab_d$pcc); r_k <- cor(tab_k$H.CV, tab_k$pcc)
  p_d <- cor.test(tab_d$H.CV, tab_d$pcc)$p.value
  p_k <- cor.test(tab_k$H.CV, tab_k$pcc)$p.value
  fmt_p <- function(p) if (p < 0.001) "<0.001" else sprintf("%.3f", p)
  cat(sprintf("  %s: tp1+Meg+Dgp r=%.3f p=%s, tp1+knode r=%.3f p=%s\n", title, r_d, fmt_p(p_d), r_k, fmt_p(p_k)))
  xr <- max(tab$H.CV, na.rm = TRUE); yr <- max(tab$pcc, na.rm = TRUE)
  xl <- min(tab$H.CV, na.rm = TRUE); yl <- min(tab$pcc, na.rm = TRUE)
  ggplot(tab, aes(x = H.CV, y = pcc, shape = Source, colour = Source)) +
    geom_point(size = 2) +
    scale_shape_manual(values = c("DynamicGP+MegaLMM+TP1" = 23, "Kode+TP1" = 21)) +
    scale_color_manual(values = c("DynamicGP+MegaLMM+TP1" = dgp_col, "Kode+TP1" = kn_col)) +
    geom_smooth(method = lm, se = FALSE, linewidth = 0.5) +
    annotate("text", x = xl + (xr - xl) * 0.38, y = yl + (yr - yl) * 0.92, size = 2, hjust = 0, fontface = "bold",
             label = sprintf("DynamicGP+MegaLMM+TP1: r=%.3f, p=%s\nKode+TP1: r=%.3f, p=%s", r_d, fmt_p(p_d), r_k, fmt_p(p_k))) +
    xlab("CV of heritability") + ylab("Prediction accuracy") + ggtitle(title) +
    theme_classic(base_family = "Times New Roman", base_size = 8) +
    theme(legend.position = "top", text = element_text(family = "Times New Roman"),
          legend.title = element_text(size = 6, face = "bold"),
          legend.text = element_text(size = 6, face = "bold"),
          axis.title = element_text(size = title_size, face = "bold"),
          axis.text = element_text(size = axis_text_size, face = "bold"),
          title = element_text(size = 9, face = "bold"))
}
d_ <- make_cv(pa_tr[pa_tr$method %in% c("tp1+Meg+Dgp_iter", "tp1+knode_iter"), ], "d (iter)", "#1F77B4", "#D62728")
e_ <- make_cv(pa_tr[pa_tr$method %in% c("tp1+Meg+Dgp_rec", "tp1+knode_rec"), ], "e (rec)", "#1F77B4", "#D62728")

layout <- "
AB
CD
EF
"
multi <- a_iter + a_rec + b + c_ + d_ + e_ + plot_layout(design = layout)
ggsave(file.path(HERE, sprintf("FigureE3_%s.svg", ds)), multi, device = "svg",
       width = 200, height = 200, units = "mm")
cat(sprintf("Saved: %s/FigureE3_%s.svg\n", HERE, ds))
