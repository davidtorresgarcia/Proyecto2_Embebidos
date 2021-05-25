# Recipe created by recipetool
# This is the basis of a recipe and may need further editing in order to be ful$
# (Feel free to remove these comments when editing.)
# Unable to find any files that looked like license statements. Check the accom$
# documentation and source headers and set LICENSE and LIC_FILES_CHKSUM accordi$
# NOTE: LICENSE is being set to "CLOSED" to allow you to at least start buildin$
# this is not accurate with respect to the licensing of the software being buil$
# will not be in most cases) you must specify the correct value before using th$
# recipe for anything other than initial testing/development!

LICENSE = "CLOSED"
LIC_FILES_CHKSUM = ""

# No information for SRC_URI yet (only an external source tree was specified)

SRC_URI = "file://programapy-1.0.tar.gz"

# NOTE: no Makefile found, unable to determine what needs to be done

S = "${WORKDIR}"

TARGET_CC_ARCH += "${LDFLAGS}"

do_configure () {
    	# Specify any needed configure commands here
    	:
}

do_install () {
    install -d ${D}${bindir}
    install -m 0755 emotion.py ${D}${bindir}
	install -m 0755 fer.h5 ${D}${bindir}
	install -m 0755 fer.json ${D}${bindir}
	install -m 0755 haarcascade_frontalface_default.xml ${D}${bindir}
	install -m 0755 happy.png ${D}${bindir}
}
